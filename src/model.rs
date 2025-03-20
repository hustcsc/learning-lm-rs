use std::fs::File;
use std::vec;

use crate::config::LlamaConfigJson;
use crate::kvcache::KVCache;
use crate::operators as OP;
use crate::operators::*;
use crate::params::LLamaParams;
use crate::tensor::Tensor;
use safetensors::SafeTensors;
use std::path::Path;


#[allow(dead_code)]
pub struct Llama<T> {
    vocab: usize,           // vocab size
    n_layers: usize,        // number of layers
    n_q_h: usize,           // number of heads for q
    n_kv_h: usize,          // number of heads for k and v
    d: usize,               // dimension of hidden states
    dqkv: usize,            // length of a single q, k, or v vector
    di: usize,              // dimension of intermediate states
    eps: f32,               // epsilon for RMS normalization
    rope_theta: f32,        // rope theta for rope initialization
    max_seq_len: usize,     // maximum sequence length
    params: LLamaParams<T>, // trained weights of this model
    bos_token_id: u32,      // start token id
    eos_token_id: u32,      // end token id
}

impl Llama<f32> {
    pub fn from_safetensors(model_dir: impl AsRef<Path>) -> Self {
        let config = File::open(model_dir.as_ref().join("config.json")).unwrap();
        let config: LlamaConfigJson = serde_json::from_reader(config).unwrap();
        let model_file = std::fs::read(model_dir.as_ref().join("model.safetensors")).unwrap();
        let safetensor = SafeTensors::deserialize(&model_file).unwrap();
        let params = LLamaParams::from_safetensors(&safetensor, &config);

        Self {
            vocab: config.vocab_size,
            n_layers: config.num_hidden_layers,
            n_q_h: config.num_attention_heads,
            n_kv_h: config.num_key_value_heads,
            d: config.hidden_size,
            dqkv: config.hidden_size / config.num_attention_heads,
            di: config.intermediate_size,
            eps: config.rms_norm_eps,
            rope_theta: config.rope_theta,
            max_seq_len: config.max_position_embeddings,
            params: params,
            bos_token_id: config.bos_token_id,
            eos_token_id: config.eos_token_id,
        }
    }

    pub fn new_cache(&self) -> KVCache<f32> {
        KVCache::new(self.n_layers, self.max_seq_len, self.n_kv_h * self.dqkv, 0)
    }

    pub fn forward(&self, input: &Tensor<u32>, cache: &mut KVCache<f32>) -> Tensor<f32> {
        let seq_len = input.size();
        let past_seq_len = cache.len();
        cache.increment(seq_len);
        let total_seq_len = past_seq_len + seq_len;
        let n_groups = self.n_q_h / self.n_kv_h;

        // Some pre-allocated buffers that will be reused
        let mut residual = Tensor::<f32>::default(&vec![seq_len, self.d]);
        let mut hidden_states = Tensor::<f32>::default(&vec![seq_len, self.d]);
        let mut q_buf = Tensor::<f32>::default(&vec![seq_len, self.n_q_h * self.dqkv]);
        let mut att_scores =
            Tensor::<f32>::default(&vec![self.n_kv_h, n_groups, seq_len, total_seq_len]);
        let mut gate_buf = Tensor::<f32>::default(&vec![seq_len, self.di]);
        let mut up_buf = Tensor::<f32>::default(&vec![seq_len, self.di]);

        // Computation Starts Here
        // Embedding lookup
        OP::gather(&mut residual, input, &self.params.embedding_table);

        for layer in 0..self.n_layers {
            OP::rms_norm(
                &mut hidden_states,
                &residual,
                &self.params.rms_att_w[layer],
                self.eps,
            );

            let q = (&mut q_buf).reshape(&vec![seq_len, self.n_q_h * self.dqkv]); // (seq, n_h * dqkv)
            let k = &mut cache.k_cache(layer, past_seq_len); // (seq, n_kv_h * dqkv)
            let v = &mut cache.v_cache(layer, past_seq_len); // (seq, n_kv_h * dqkv)
            OP::matmul_transb_avx(q, 0., &hidden_states, &self.params.wq[layer], 1.0);
            OP::matmul_transb_avx(k, 0., &hidden_states, &self.params.wk[layer], 1.0);
            OP::matmul_transb_avx(v, 0., &hidden_states, &self.params.wv[layer], 1.0);
            OP::rope(
                q.reshape(&vec![seq_len, self.n_q_h, self.dqkv]),
                past_seq_len,
                self.rope_theta,
            );
            OP::rope(
                k.reshape(&vec![seq_len, self.n_kv_h, self.dqkv]),
                past_seq_len,
                self.rope_theta,
            );

            let full_k = &mut cache.k_cache(layer, 0); // (total_seq, n_kv_h * dqkv)
            let full_v = &mut cache.v_cache(layer, 0); // (total_seq, n_kv_h * dqkv)

            self_attention(
                &mut hidden_states,
                &mut att_scores,
                q,
                full_k,
                full_v,
                self.n_kv_h,
                n_groups,
                seq_len,
                total_seq_len,
                self.dqkv,
            );

            // out = attn_V @ O_weight.T
            // residual = out + residual
            OP::matmul_transb_avx(
                &mut residual,
                1.0,
                &hidden_states,
                &self.params.wo[layer],
                1.0,
            );

            mlp(
                &mut residual,
                &mut hidden_states,
                &mut gate_buf,
                &mut up_buf,
                &self.params.w_up[layer],
                &self.params.w_down[layer],
                &self.params.w_gate[layer],
                &self.params.rms_ffn_w[layer],
                self.eps,
            );
        }


        // No matter what seq_len, the output is always a 1D vector of length vocab,
        // which contains the probabilities for the next token.
        let mut logits = Tensor::<f32>::default(&vec![1, self.vocab]);
        // println!("{}",5);
        let mut hidden_states = hidden_states.slice((seq_len - 1) * self.d, &vec![1, self.d]);
        let residual = residual.slice((seq_len - 1) * self.d, &vec![self.d]);

        OP::rms_norm(
            &mut hidden_states,
            &residual,
            &self.params.rms_out_w,
            self.eps,
        );

        OP::matmul_transb(&mut logits, 0., &hidden_states, &self.params.lm_head, 1.0);

        logits
    }


    pub fn generate(
        &self,
        token_ids: &[u32],
        max_len: usize,
        top_p: f32,
        top_k: u32,
        temperature: f32,
    ) -> Vec<u32> {
        let mut result = token_ids.to_vec();
        
        if result.is_empty() {
            result.push(self.bos_token_id);
        }

        let mut cache = self.new_cache();
        let prompt_tensor = Tensor::<u32>::new(result.clone(), &vec![result.len()]);
        let _ = self.forward(&prompt_tensor, &mut cache);

        for _ in 0..max_len {
            let last_token = *result.last().unwrap();
            let input_tensor = Tensor::<u32>::new(vec![last_token], &vec![1]);
            let logits = self.forward(&input_tensor, &mut cache);
            let next_token = OP::random_sample(&logits, top_p, top_k, temperature);
            result.push(next_token);

            if next_token == self.eos_token_id {
                break;
            }
        }

        result
    }



    // pub fn chat_generate1<'a>(
    //     &'a self,
    //     token_ids: &[u32],
    //     max_len: usize,
    //     top_p: f32,
    //     top_k: u32,
    //     temperature: f32,
    //     kvcache: &'a mut KVCache<f32>,
    // ) -> impl Iterator<Item = u32> + 'a {

    //     let mut result = Vec::<u32>::new();
    //     let mut input = Tensor::<u32>::new(Vec::from(token_ids), &vec![1, result.len()]);

    //     std::iter::from_fn(move ||  {
    //         let output = OP::random_sample(&self.forward(&input, kvcache), top_p, top_k, temperature);
    //         result.push(output);
    //         input = Tensor::<u32>::new(Vec::from([output]), &vec![1, 1]);

    //         if result.len() >= max_len || output == self.eos_token_id {
    //             None
    //         } else {
    //             Some(output)
    //         }
    //     })
    // }



    pub fn chat_generate(
        &self,
        token_ids: &[u32],
        max_len: usize,
        top_p: f32,
        top_k: u32,
        temperature: f32,
        kvcache: &mut KVCache<f32>,
    ) -> Vec<u32> {
        let mut result = vec![]; // 结果列表，初始化为 token_ids
    
        while result.len() <= max_len { // 避免超过 max_len
            // 生成输入 token：第一步使用 token_ids，之后每次只输入上一个 token
            let input_tokens = if result.len() == 0 {
                token_ids.to_vec() // 初始 prompt 作为输入
            } else {
                vec![*result.last().unwrap()] // 之后每次只输入上一个 token
            };
            let len = input_tokens.len();
            let input = Tensor::<u32>::new(input_tokens, &vec![len]);
    
            // 执行前向计算
            let logits = self.forward(&input, kvcache);
            
            // 采样得到新 token
            let id = OP::random_sample(&logits, top_p, top_k, temperature);
            result.push(id);
    
            // 终止条件：如果生成了 EOS（假设 EOS = 0）
            if id == self.eos_token_id { 
                break;
            }
        }
        result.pop(); // 去掉 EOS
        result        
        // todo!("Add new function to attach the model to the chatbot");
    }






}




fn self_attention(
    hidden_states: &mut Tensor<f32>, // (seq, n_kv_h * n_groups * dqkv)
    att_scores: &mut Tensor<f32>,    // (n_kv_h, n_groups, seq, total_seq)
    q: &Tensor<f32>,                 // (seq, n_kv_h * n_groups * dqkv)
    k: &Tensor<f32>,                 // (total_seq, n_kv_h * dqkv)
    v: &Tensor<f32>,                 // (total_seq, n_kv_h * dqkv) or (n_kv_h * dqkv, total_seq)
    n_kv_h: usize,
    n_groups: usize,
    seq_len: usize,
    total_seq_len: usize,
    dqkv: usize,
) {
    assert!(hidden_states.size() == q.size());
    assert!(att_scores.size() == n_kv_h*n_groups*seq_len*total_seq_len);
    let n_q_h = n_kv_h * n_groups;
    let q_dim = n_q_h * dqkv;
    let kv_dim = n_kv_h * dqkv;
    let att_dim = n_q_h * total_seq_len;
    let q_data = q.data();
    let k_data = k.data();
    //转置v矩阵，shape为(n_kv_h * dqkv, total_seq_len)
    let v = v.transpose();
    let v_data = v.data();
    let att_data = unsafe { att_scores.data_mut() };
    let hidden_data = unsafe { hidden_states.data_mut() };
    
    // 计算attention矩阵
    // attention的shape为(seq_len, n_q_h * total_seq_len)
    // query矩阵的每个head的shape为(seq_len, dqkv)
    // key矩阵的每个head的shape为(total_seq_len, dqkv)
    for h in 0..n_q_h {
        for row in 0..seq_len {
            for col in 0..total_seq_len {
                let q_base = row * q_dim + h * dqkv;
                let k_base = col * kv_dim + h/n_groups * dqkv;
                att_data[row * att_dim + h * total_seq_len + col] = (0..dqkv).map(|d| q_data[q_base + d] * k_data[k_base + d]).sum::<f32>() / (dqkv as f32).sqrt();
            }
        }
    }
    OP::masked_softmax(att_scores);
     
    let att = att_scores.data(); 
    // attention的shape为(seq_len, n_q_h * total_seq_len)
    // v的shape为(n_kv_h * dqkv, total_seq_len)
    // attention的每个head的shape为(seq_len, total_seq_len)
    //
    for h in 0..n_q_h {
        for row in 0..seq_len {
            for col in 0..dqkv {
                let a_base = row * att_dim + h * total_seq_len;
                //let v_base = col + h/n_groups * dqkv;
                let v_base = col * total_seq_len + h/n_groups * dqkv * total_seq_len;
                hidden_data[row * q_dim + h * dqkv + col] = (0..total_seq_len).map(|d| att[a_base + d] * v_data[v_base + d]).sum::<f32>();
            }
        }
    } 
}



fn self_attention1(
    hidden_states: &mut Tensor<f32>, // (seq, n_kv_h * n_groups * dqkv)
    att_scores: &mut Tensor<f32>,    // (n_kv_h, n_groups, seq, total_seq)
    q: &Tensor<f32>,                 // (seq, n_kv_h * n_groups * dqkv)
    k: &Tensor<f32>,                 // (total_seq, n_kv_h * dqkv)
    v: &Tensor<f32>,                 // (total_seq, n_kv_h * dqkv)
    n_kv_h: usize,
    n_groups: usize,
    seq_len: usize,
    total_seq_len: usize,
    dqkv: usize,
) {
    // todo!("Implement self_attention");
    let _a = unsafe { att_scores.data_mut() };
    let _q = q.data();
    let _k = k.data();
    let _v = v.data();
    let sqrt_dim = (dqkv as f32).sqrt();

    // score = Q @ K.T / sqrt(dim)
    // (seq_len, n_kv_h * n_groups * dqkv) @ (total_seq, n_kv_h * dqkv).T
    for h in 0..(n_kv_h * n_groups) {
        for q_seq in 0..seq_len {
            for k_seq in 0..total_seq_len {
                let sum = (0..dqkv)
                        .map(|j| {
                        _q[q_seq * n_kv_h * n_groups * dqkv + h * dqkv + j]
                            * _k[k_seq * n_kv_h * dqkv + h / n_groups * dqkv + j]
                        })
                        .sum::<f32>();
                
                _a[h * seq_len * total_seq_len + q_seq * total_seq_len + k_seq] = sum / sqrt_dim;          
            }
        }
    }

    // attn = softmax(score)
    OP::masked_softmax(att_scores);

    // attn_V = attn @ V
    let _a = att_scores.data();
    let _h = unsafe { hidden_states.data_mut() };
    for h in 0..(n_kv_h * n_groups) {
        for q_seq in 0..seq_len {
            for i in 0..dqkv {
                let sum = (0..total_seq_len)
                        .map(|k_seq| {
                            _a[h * seq_len * total_seq_len + q_seq * total_seq_len + k_seq]
                                * _v[i + h / n_groups * dqkv + k_seq * n_kv_h * dqkv]
                        })
                        .sum::<f32>();
                _h[q_seq * n_kv_h * n_groups * dqkv + h * dqkv + i] = sum;
            }
        }
    } 
}

fn mlp(
    residual: &mut Tensor<f32>,
    hidden_states: &mut Tensor<f32>,
    gate: &mut Tensor<f32>,
    up: &mut Tensor<f32>,
    w_up: &Tensor<f32>,
    w_down: &Tensor<f32>,
    w_gate: &Tensor<f32>,
    rms_w: &Tensor<f32>,
    eps: f32,
) {
    // todo!("Implement mlp");
    rms_norm(hidden_states, residual, rms_w, eps);
    matmul_transb(gate,0., hidden_states, w_gate, 1.);
    matmul_transb(up, 0., hidden_states,w_up, 1.);
    swiglu(up, gate);
    matmul_transb(residual,1.,up,w_down,1.);



}

#[test]
pub fn test_mlp() {
    let seq_len = 4;
    let d = 2;
    let di = 3;
    let mut residual = Tensor::<f32>::new(vec![1., 1., 1., 1., 1., 1., 1., 1.], &vec![seq_len, d]);
    let mut hidden_states = Tensor::<f32>::default(&vec![seq_len, d]);
    let mut gate_buf = Tensor::<f32>::default(&vec![seq_len, di]);
    let mut up_buf = Tensor::<f32>::default(&vec![seq_len, di]);
    let w_up = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![di, d]);
    let w_down = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![d, di]);
    let w_gate = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![di, d]);
    let rms_w = Tensor::<f32>::new(vec![1., 1.], &vec![d]);
    let eps = 1e-6;
    mlp(
        &mut residual,
        &mut hidden_states,
        &mut gate_buf,
        &mut up_buf,
        &w_up,
        &w_down,
        &w_gate,
        &rms_w,
        eps,
    );

    assert!(residual.close_to(
        &Tensor::<f32>::new(
            vec![
                1.3429964, 1.7290739, 1.3429964, 1.7290739, 1.3429964, 1.7290739, 1.3429964,
                1.7290739
            ],
            &vec![seq_len, d]
        ),
        1e-3
    ))
}

#[test]
pub fn test_load_safetensors() {
    use std::path::PathBuf;
    use crate::tensor::float_eq;
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("story");
    let model = Llama::from_safetensors(model_dir);
    assert_eq!(model.vocab, 2048);
    assert_eq!(model.n_layers, 2);
    assert_eq!(model.n_q_h, 8);
    assert_eq!(model.n_kv_h, 4);
    assert_eq!(model.d, 128);
    assert_eq!(model.dqkv, 16);
    assert_eq!(model.di, 384);

    assert!(float_eq(&model.params.embedding_table.data()[50], &0.14453125, 1e-6));
    assert_eq!(model.params.lm_head.data()[10], model.params.embedding_table.data()[10]);
    assert!(float_eq(&model.params.rms_att_w[0].data()[10], &0.18652344, 1e-6));
    assert!(float_eq(&model.params.rms_ffn_w[1].data()[10], &0.32421875, 1e-6));
    assert!(float_eq(&model.params.rms_out_w.data()[100], &0.73046875, 1e-6));
    assert!(float_eq(&model.params.w_down[0].data()[100], &-0.0625, 1e-6));
    assert!(float_eq(&model.params.w_up[0].data()[100], &1.46875, 1e-6));
    assert!(float_eq(&model.params.w_gate[1].data()[100], &0.296875, 1e-6));
    assert!(float_eq(&model.params.wq[1].data()[100], &0.032226563, 1e-6));
    assert!(float_eq(&model.params.wk[1].data()[100], &-0.21386719, 1e-6));
    assert!(float_eq(&model.params.wv[0].data()[100], &0.041015625, 1e-6));
    assert!(float_eq(&model.params.wo[0].data()[100], &0.01965332, 1e-6));

}

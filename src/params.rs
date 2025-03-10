use crate::config::LlamaConfigJson;
use crate::tensor::Tensor;
use safetensors::SafeTensors;
pub struct LLamaParams<T> {
    // token_id to embedding lookup table
    pub embedding_table: Tensor<T>, // (vocab_size, dim)
    // decoder layer
    pub rms_att_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub wq: Vec<Tensor<T>>,        // (n_heads * head_size, hidden_size) x layers
    pub wk: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wv: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wo: Vec<Tensor<T>>,        // (hidden_size, n_heads * head_size) x layers
    // ffn layer
    pub rms_ffn_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub w_up: Vec<Tensor<T>>,      // (intermediate_size, hidden_size) x layers
    pub w_gate: Vec<Tensor<T>>,    // (intermediate_size, hidden_size) x layers
    pub w_down: Vec<Tensor<T>>,    // (hidden_size, intermediate_size) x layers
    // output
    pub rms_out_w: Tensor<T>, // (hidden_size, )
    pub lm_head: Tensor<T>,   // (vocab_size, dim)
}

impl LLamaParams<f32> {
    pub fn from_safetensors(safetensor: &SafeTensors, config: &LlamaConfigJson) -> Self {
<<<<<<< HEAD
        // todo!("实现从safetensors文件的模型参数加载");
        // let get_tensor: impl Fn(&str) -> Tensor<f32> = |name: &str| {
        // ...    
        // };
        // error[E0562]: `impl Trait` is not allowed in the type of variable bindings
        // note: `impl Trait` is only allowed in arguments and return types of functions and methods
        
        // LLamaParams {
        //     embedding_table: get_tensor(...),
        //     ...
        // }

        let get_tensor = |name: &str| -> Tensor<f32> {
            let tensor = safetensor.tensor(name).unwrap();
            let shape = tensor.shape().to_vec();
            let data: Vec<f32> = tensor.data().chunks_exact(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                .collect();
            // fn new(data: Vec<T>, shape: &Vec<usize>) 
            Tensor::new(data, &shape)
        };

        let n_layers = config.num_hidden_layers;
        LLamaParams {
            // token_id to embedding lookup table
            embedding_table: if config.tie_word_embeddings {
                get_tensor("lm_head.weight")
            } else {
                get_tensor("model.embed_tokens.weight")
            },
            // decoder layer
            rms_att_w: (0..n_layers)
                .map(|layer| get_tensor(&format!("model.layers.{layer}.input_layernorm.weight")))
                .collect(),
            wq: (0..n_layers)
                .map(|layer| get_tensor(&format!("model.layers.{layer}.self_attn.q_proj.weight")))
                .collect(),
            wk: (0..n_layers)
                .map(|layer| get_tensor(&format!("model.layers.{layer}.self_attn.k_proj.weight")))
                .collect(),
            wv: (0..n_layers)
                .map(|layer| get_tensor(&format!("model.layers.{layer}.self_attn.v_proj.weight")))
                .collect(),
            wo: (0..n_layers)
                .map(|layer| get_tensor(&format!("model.layers.{layer}.self_attn.o_proj.weight")))
                .collect(),
            // ffn layer
            rms_ffn_w: (0..n_layers)
                .map(|layer| get_tensor(&format!("model.layers.{layer}.post_attention_layernorm.weight")))
                .collect(),
            w_up: (0..n_layers)
                .map(|layer| get_tensor(&format!("model.layers.{layer}.mlp.up_proj.weight")))
                .collect(),
            w_gate: (0..n_layers)
                .map(|layer| get_tensor(&format!("model.layers.{layer}.mlp.gate_proj.weight")))
                .collect(),
            w_down: (0..n_layers)
                .map(|layer| get_tensor(&format!("model.layers.{layer}.mlp.down_proj.weight")))
                .collect(),
            // output
=======
        let get_tensor = |name: &str| -> Tensor<f32> {
            let tensor = safetensor.tensor(name).unwrap();
            let shape = tensor.shape();
            let raw_data = tensor.data();
            let len: usize = shape.iter().clone().product();
            let mut data: Vec<f32> = vec![];
            if raw_data.len() / std::mem::size_of::<f32>() == len {
                data = raw_data
                    .chunks_exact(std::mem::size_of::<f32>())
                    .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                    .collect();
            }
            Tensor::new(data, &shape.to_vec())
        };

        LLamaParams {
            embedding_table: get_tensor("lm_head.weight"),
            rms_att_w: (0..config.num_hidden_layers)
                .map(|i| get_tensor(&format!("model.layers.{}.input_layernorm.weight", i)))
                .collect(),
            wq: (0..config.num_hidden_layers)
                .map(|i| get_tensor(&format!("model.layers.{}.self_attn.q_proj.weight", i)))
                .collect(),
            wk: (0..config.num_hidden_layers)
                .map(|i| get_tensor(&format!("model.layers.{}.self_attn.k_proj.weight", i)))
                .collect(),
            wv: (0..config.num_hidden_layers)
                .map(|i| get_tensor(&format!("model.layers.{}.self_attn.v_proj.weight", i)))
                .collect(),
            wo: (0..config.num_hidden_layers)
                .map(|i| get_tensor(&format!("model.layers.{}.self_attn.o_proj.weight", i)))
                .collect(),
            rms_ffn_w: (0..config.num_hidden_layers)
                .map(|i| get_tensor(&format!("model.layers.{}.post_attention_layernorm.weight", i)))
                .collect(),
            w_up: (0..config.num_hidden_layers)
                .map(|i| get_tensor(&format!("model.layers.{}.mlp.up_proj.weight", i)))
                .collect(),
            w_gate: (0..config.num_hidden_layers)
                .map(|i| get_tensor(&format!("model.layers.{}.mlp.gate_proj.weight", i)))
                .collect(),
            w_down: (0..config.num_hidden_layers)
                .map(|i| get_tensor(&format!("model.layers.{}.mlp.down_proj.weight", i)))
                .collect(),
>>>>>>> cd422c626056dcdfc5984b50874ee6a3c7820a3a
            rms_out_w: get_tensor("model.norm.weight"),
            lm_head: get_tensor("lm_head.weight"),
        }
    }
}
#![allow(unused_imports, dead_code)]
mod config;
mod kvcache;
mod model;
mod operators;
mod params;
mod tensor;

use std::path::PathBuf;
use tokenizers::Tokenizer;
use std::io::stdin;

use std::env;
// use std::path::PathBuf;
use std::io::{self, BufRead};
use model::Llama;

struct Prompt {
    // history: String,
    system_message: String,
    user_message: String,
}
// 定义 Prompt 结构体
pub fn story() {
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("story");
    let llama = model::Llama::<f32>::from_safetensors(&model_dir);
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
    let input = "Once upon a time";
    let binding = tokenizer.encode(input, true).unwrap();
    let input_ids = binding.get_ids();
    print!("\n{}", input);
    let output_ids = llama.generate(
        input_ids,
        500,
        0.8,
        30,
        1.,
    );
    println!("{}", tokenizer.decode(&output_ids, true).unwrap());
}



pub fn chat(){
    let project_dir = env!("CARGO_MANIFEST_DIR");
    // let model_dir = PathBuf::from(project_dir).join("models").join("story");
    let model_dir = PathBuf::from(project_dir).join("models").join("chat");
    let llama =  model::Llama::<f32>::from_safetensors(&model_dir);

    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();

    let mut prompt = Prompt {
        // history: String::new(),
        system_message: "You are a helpful assistant.".to_string(),
        user_message: String::new(),
    };
    let mut kvcache = llama.new_cache();
    // kvcache = kvcache.as_mut().unwrap();
    loop {
        // 获取用户输入
        println!("User: ");
        let mut user_input = String::new();
        std::io::stdin().read_line(&mut user_input).unwrap();
        prompt.user_message = user_input.trim().to_string();

        // 构建输入字符串
        let input = format!("<|im_start|>system\n{0}<|im_end|>\n<|im_start|>user\n{1}<|im_end|>\n<|im_start|>assistant",
             prompt.system_message, prompt.user_message
        );

        // 编码输入
        let binding = tokenizer.encode(input, true).unwrap();
        let input_ids = binding.get_ids();

        // 生成输出
        // let output_ids = llama.generate(
        //     input_ids,
        //     200,  // max_length
        //     0.9,  // temperature
        //     30,   // top_k
        //     1.0,  // top_p
        // );
        let output_ids: Vec<u32> = llama.chat_generate(input_ids, 500, 0.8, 30, 1., &mut kvcache).collect();

        // 解码输出
        let assistant_response = tokenizer.decode(&output_ids, true).unwrap();
        println!("Assistant: {}", assistant_response);

        // 更新历史记录
        // prompt.history = format!(
        //     "<|im_start|>user\n{0}<|im_end|>\n<|im_start|>assistant\n{1}<|im_end|>\n",
        //      prompt.user_message, assistant_response
        // );
    }
}


fn main() {
    // chat();
        println!("Please enter a number:");
        println!("1. Chat");
        println!("2. Story");
        println!("Enter any other number or non - numeric input to exit.");

        let mut input = String::new();
        match stdin().read_line(&mut input) {
            Ok(_) => {
                let input = input.trim();
                match input.parse::<u32>() {
                    Ok(num) => {
                        match num {
                            1 => chat(),
                            2 => story(),
                            _ => {
                                println!("Exiting the program.");
                                // break;
                            }
                        }
                    }
                    Err(_) => {
                        println!("Exiting the program due to non - numeric input.");
                        // break;
                    }
                }
            }
            Err(_) => {
                println!("Failed to read input. Exiting the program.");
                // break;
            }
        }
    
}



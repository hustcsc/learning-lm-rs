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

// struct Prompt {
//     // history: String,
//     system_message: String,
//     user_message: String,
// }
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
    let model_dir = PathBuf::from(project_dir).join("models").join("story");
    let llama =  model::Llama::<f32>::from_safetensors(&model_dir);

    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();

    // let mut prompt = Prompt {
    //     // history: String::new(),
    //     system_message: "You are a helpful assistant.".to_string(),
    //     user_message: String::new(),
    // };
    let mut prompt = String::new();
    let mut kvcache = llama.new_cache();
    
    // let mut vec = Vec::new();
    // 可以向 vec 中添加一些元素
    // vec.push(("Hello".to_string(), "World".to_string()));
    // 获取 vec 的可变引用，类型为 &mut Vec<(String, String)>
    let mut history =  Vec::<(String, String)>::new();

    // 现在可以通过 history 对 vec 进行操作
    // history.push(("How are you?".to_string(), "I am fine".to_string()));
    // kvcache = kvcache.as_mut().unwrap();
    loop {
        println!("\n\n#####################请输入聊天的模式#################");
        println!("1. 普通聊天模式");
        println!("2. 历史记录模式");
        println!("3. 退出\n\n");

        let mut input = String::new();
        match stdin().read_line(&mut input) {
            Ok(_) => {
                let input = input.trim();
                match input.parse::<u32>() {
                    Ok(num) => {
                        match num {
                            1 => {// 获取用户输入
                                println!("User: ");
                                let mut user_input = String::new();
                                std::io::stdin().read_line(&mut user_input).unwrap();
                                // prompt.user_message = user_input.trim().to_string();
                                for (user, assistant) in history.iter() {
                                    prompt.push_str(&format!("<|im_start|>user\n{}<|im_end|>\n", user));
                                    prompt.push_str(&format!("<|im_start|>assistant\n{}\n<|im_end|>\n", assistant));


                                    // prompt.push_str(&format!("<|im_start|>user\n{}<|im_end|>\n", user));
                                    // prompt.push_str(&format!("<|im_start|>assistant\n{}\n<|im_end|>\n", assistant));
                                }
                                
                                // 正确格式化新用户输入，添加换行符
                                prompt.push_str("<|im_start|>system\nyou are a helpfull assistant<|im_end|>\n"); // 让模型开始回答mpt.push_str("<|im_start|>assistant"); // 让模型开始回答
                                prompt.push_str("<|im_start|>user\nwho are you<|im_end|>\n");
                                prompt.push_str("<|im_start|>assistant"); // 让模型开始回答mpt.push_str("<|im_start|>assistant"); // 让模型开始回答
                                

                                // 构建输入字符串
                                // let input = format!("<|im_start|>system\n{0}<|im_end|>\n<|im_start|>user\n{1}<|im_end|>\n<|im_start|>assistant",
                                //      prompt.system_message, prompt.user_message
                                // );
                                // let input = format!("<|im_start|>system\n{}<|im_end|>\n{}",prompt.system_message, prompt.user_message);
                                // 编码输入
                                let binding = tokenizer.encode(prompt.as_str(), true).unwrap();
                                let input_ids = binding.get_ids();
                        
                                // 生成输出
                                // let output_ids = llama.generate(
                                //     input_ids,
                                //     200,  // max_length
                                //     0.9,  // temperature
                                //     30,   // top_k
                                //     1.0,  // top_p
                                // );
                                let output_ids: Vec<u32> = llama.chat_generate(input_ids, 500, 0.9, 30, 1., &mut kvcache).into_iter().collect();
                                // let output_ids : Vec<u32> = llama.generate(input_ids, 500, 0.8, 30, 1.0); // 保持原参数
                                // // 解码输出
                                // let assistant_response = tokenizer.decode(&output_ids, true).unwrap();
                                let assistant_response = tokenizer.decode(&output_ids, true).unwrap();
                                let assistant_response = assistant_response.replace("<|end_story|>", "").trim().to_string();
                                   // 7. 记录对话历史
                                history.push((user_input.trim().to_string(), assistant_response.clone()));

                                println!("Assistant: {}", assistant_response);
                                
                            },
                            2 => {
                                println!("\n#####################History###########################");
                                for (user, assistant) in history.iter() {
                                    println!("User: {},\nAssistant: {}", user, assistant);
                                } 
                                println!("#####################History###########################\n");                            
                            },
                            _ => {
                                println!("Exiting the program.");
                                break;
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





        
    

        // 更新历史记录
        // prompt.history = format!(
        //     "<|im_start|>user\n{0}<|im_end|>\n<|im_start|>assistant\n{1}<|im_end|>\n",
        //      prompt.user_message, assistant_response
        // );
    }
}


fn main() {
    // chat();
        println!("\n\n\n\n######################################");
        println!("欢迎来到我的AI大语言模型！");
        println!("######################################");
        println!("请选择功能：");
        println!("1. Chat(请输入英文)");
        println!("2. Story \n\n\n");

        // println!("Enter any other number or non - numeric input to exit.");

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



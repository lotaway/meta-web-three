use std::{env, fs, path::Path};

fn main() {
    println!("Start rust building script...");
    let mut proto_files = vec![];
    let out_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let proto_dir = Path::new(&out_dir).join("../protos/");
    if let Ok(entries) = fs::read_dir(proto_dir.clone()) {
        for entry in entries {
            if let Ok(entry) = entry {
                let path = entry.path();
                println!("Checking path: {}", path.to_str().unwrap());
                if path.is_file() && path.extension().unwrap_or_default() == "proto" {
                    proto_files.push(path.to_str().unwrap().to_string());
                }
            }
        }
    } else {
        panic!("Failed to read directory: {:?}", proto_dir.to_str().unwrap().to_string());
    }

    match tonic_prost_build::configure()
        .build_client(true)
        .build_server(true)
        .out_dir("src/generated/rpc/")
        .compile_protos(&proto_files, &["../protos".to_string()])
    {
        Ok(data) => {
            println!("cargo:rerun-if-changed=../protos/*.proto");
            data
        }
        Err(e) => {
            panic!("Failed to compile protos: {}", e);
        }
    }
}

use std::fs;

fn main() {
    let mut proto_files = vec![];
    let proto_dir = "../protos/";
    if let Ok(entries) = fs::read_dir(proto_dir) {
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
        panic!("Failed to read directory: {}", proto_dir);
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

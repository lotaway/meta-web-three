use std::{
    env, fs,
    path::{Path, PathBuf},
};

fn main() {
    // start_build();
    start_build_v2();
}

fn get_proto_dir() -> PathBuf {
    Path::new("../protos")
        .canonicalize()
        .expect("Error in canonicalize prot dir path")
}

fn get_proto_files(proto_dir: PathBuf) -> Vec<PathBuf> {
    let mut proto_files = vec![];
    println!("Checking path: {}", proto_dir.display());
    if let Ok(entries) = fs::read_dir(proto_dir.clone()) {
        for entry in entries {
            if let Ok(entry) = entry {
                let proto_file_path = entry.path();
                if proto_file_path.is_file()
                    && proto_file_path.extension().unwrap_or_default() == "proto"
                {
                    println!("Checking file path: {}", proto_file_path.clone().display());
                    // let file_name = proto_file_path
                    //     .file_name()
                    //     .expect("Error in getting file name")
                    //     .to_os_string();
                    proto_files.push(proto_file_path);
                }
            }
        }
        return proto_files;
    } else {
        panic!(
            "Failed to read directory: {:?}",
            proto_dir.to_str().unwrap().to_string()
        );
    }
}

fn start_build() {
    println!("Start rust building script...");
    // let _out_dir = "src/generated/rpc";
    // // fs::create_dir_all(_out_dir).expect("failed to create output directory");
    // let proto_dir = get_proto_dir();
    // let proto_files = get_proto_files(proto_dir);
    // match tonic_build::configure()
    //     .type_attribute(".", "#[derive(serde::Serialize, serde::Deserialize)]")
    //     .build_client(false)
    //     .out_dir(_out_dir)
    //     .compile_protos(&proto_files, &[proto_dir])
    // {
    //     Ok(data) => {
    //         println!("Compile Success!");
    //         data
    //     }
    //     Err(e) => {
    //         panic!("Failed to compile protos: {}", e);
    //     }
    // }
}

use protobuf_codegen::CodeGen;

fn start_build_v2() {
    println!("Start rust building script...");
    let _out_dir = "src/generated/rpc";
    let proto_dir = get_proto_dir();
    let proto_files = get_proto_files(proto_dir.clone());
    CodeGen::new()
        .inputs(proto_files)
        .include(proto_dir)
        .dependency(protobuf_well_known_types::get_dependency(
            "protobuf_well_known_types",
        ))
        .output_dir(_out_dir)
        .generate_and_compile()
        .unwrap();
}

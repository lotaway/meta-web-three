use std::{
    env, fs, path::{Path, PathBuf}
};

fn main() {
    start_build();
    // start_build_v2();
}

fn start_build() {
    println!("Start rust building script...");
    let _out_dir = "src/generated/rpc";
    // fs::create_dir_all(_out_dir).expect("failed to create output directory");

    let mut proto_files = vec![];
    let proto_dir = Path::new("../protos")
        .canonicalize()
        .expect("Error in canonicalize prot dir path");
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
    } else {
        panic!(
            "Failed to read directory: {:?}",
            proto_dir.to_str().unwrap().to_string()
        );
    }
    // let files = proto_files
    //     .into_iter()
    //     .map(|p| Path::new(p.to_str().unwrap()).to_path_buf())
    //     .collect::<Vec<PathBuf>>();
    match tonic_build::configure()
        .build_client(false)
        .out_dir(_out_dir)
        .compile_protos(&proto_files, &[proto_dir])
    {
        Ok(data) => {
            println!("Compile Success!");
            data
        }
        Err(e) => {
            panic!("Failed to compile protos: {}", e);
        }
    }
}

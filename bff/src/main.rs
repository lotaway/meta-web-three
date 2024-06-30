use tg_bot::run;

#[tokio::main]
async fn main() {
    println!("Hello, world!");
    println!("Start run bot");
    run().await;
    println!("End run bot");
}

fn run_web() {
    let listener =
        std::net::TcpListener::bind("127.0.0.1:4000").expect("Failed to start web server");
    for stream in listener.incoming() {
        let staam = stream.unwrap();
        println!("stream:{}", stream)
    }
}

fn handle_connection(mut stream: std::net::TcpStream) {
    let buf_reader = std::io::BufReader::new(&mut stream);
    let http_request: Vec<_> = buf_reader
        .lines()
        .map(|result| result.unwrap())
        .take_while(|line| !line.is_empty())
        .collect();
    println!("Request: {:#?}", http_request);

    let response = "HTTP/1.1 200 OK\r\n\r\n";
    stream.write_all(response.as_bytes()).unwrap();
}

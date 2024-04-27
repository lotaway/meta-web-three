# Required

* [rust-rustup](https://www.rust-lang.org/tools/install)
* [wasm-pack](https://rustwasm.github.io/wasm-pack/installer/)

## Debug

run:

```shell
wasm-pack build --target web --debug
```

or 

```shell
carge build --target web
```

or add this to Cargo.toml

```toml
[profile.release]
debug = true
```

## build

run:

```shell
wasm-pack build
```
package com.metawebthree.client;

@FeignClient(name = "product-service")
public interface ProductClient {
  @GetMapping("/product/micro-service-test")
  String microServiceTest();
}
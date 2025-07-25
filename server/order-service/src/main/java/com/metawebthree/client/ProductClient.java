package com.metawebthree.client;

import org.springframework.cloud.openfeign.FeignClient;
import org.springframework.web.bind.annotation.GetMapping;

@FeignClient(name = "product-service")
public interface ProductClient {
  @GetMapping("/product/micro-service-test")
  String microServiceTest();
}
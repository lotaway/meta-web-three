package com.metawebthree.order;

import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.*;

import com.metawebthree.client.ProductClient;

@Slf4j
@RestController
@RequestMapping("/order")
public class OrderController {

    private final ProductClient productClient;

    public OrderController(ProductClient productClient) {
        this.productClient = productClient;
    }

    @PostMapping("/create")
    public String create() {
        return "ERROR: No implementation";
    }

    @GetMapping("/micro-service-test")
    public String microServiceTest() {
        return productClient.microServiceTest();
    }

}
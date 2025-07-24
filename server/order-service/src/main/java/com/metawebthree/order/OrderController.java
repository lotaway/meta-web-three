package com.metawebthree.order;

import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.*;

@Slf4j
@RestController
@RequestMapping("/order")
public class OrderController {

    @PostMapping("/create")
    public String create() {
        return "ERROR: No implementation";
    }

}
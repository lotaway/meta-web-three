package com.metawebthree.payment.interfaces.web;

import com.metawebthree.payment.application.dto.DecisionRequest;
import com.metawebthree.payment.application.dto.DecisionResponse;
import com.metawebthree.payment.application.DecisionService;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/risk")
@RequiredArgsConstructor
public class DecisionController {

    private final DecisionService decisionService;

    @GetMapping("/test")
    public String test() {
        return String.valueOf(decisionService.test());
    }

    @PostMapping("/decision")
    public DecisionResponse decide(@RequestBody DecisionRequest request) {
        return decisionService.decide(request);
    }
}

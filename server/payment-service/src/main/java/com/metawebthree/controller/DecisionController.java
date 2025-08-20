package com.metawebthree.controller;

import com.metawebthree.dto.DecisionRequest;
import com.metawebthree.dto.DecisionResponse;
import com.metawebthree.service.DecisionService;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/risk")
@RequiredArgsConstructor
public class DecisionController {

    private final DecisionService decisionService;

    @PostMapping("/decision")
    public DecisionResponse decide(@RequestBody DecisionRequest request) {
        return decisionService.decide(request);
    }
}

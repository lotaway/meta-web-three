package com.metawebthree.service;

import com.metawebthree.dto.DecisionRequest;
import com.metawebthree.dto.DecisionResponse;

public interface DecisionService {
    int test();
    DecisionResponse decide(DecisionRequest request);
}

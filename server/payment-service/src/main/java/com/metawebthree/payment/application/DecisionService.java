package com.metawebthree.payment.application;
import com.metawebthree.payment.domain.model.*;
import com.metawebthree.payment.infrastructure.persistence.mapper.*;

import com.metawebthree.payment.application.dto.DecisionRequest;
import com.metawebthree.payment.application.dto.DecisionResponse;

public interface DecisionService {
    int test();
    DecisionResponse decide(DecisionRequest request);
}

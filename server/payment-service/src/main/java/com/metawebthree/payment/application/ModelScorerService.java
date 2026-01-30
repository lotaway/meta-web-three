package com.metawebthree.payment.application;
import com.metawebthree.payment.domain.model.*;
import com.metawebthree.payment.infrastructure.persistence.mapper.*;

import java.util.Map;

public interface ModelScorerService {
    int score(String scene, Map<String, Object> features);
}

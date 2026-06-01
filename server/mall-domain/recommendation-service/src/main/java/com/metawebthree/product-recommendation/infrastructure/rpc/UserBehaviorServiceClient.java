package com.metawebthree.product_recommendation.infrastructure.rpc;

import com.metawebthree.product_recommendation.application.dto.UserBehaviorDTO;

import java.util.ArrayList;
import java.util.List;

public class UserBehaviorServiceClient {

    public List<UserBehaviorDTO> getUserBehaviors(Long userId) {
        return new ArrayList<>();
    }

    public void recordBehavior(UserBehaviorDTO behavior) {
    }
}
package com.metawebthree.payment.infrastructure.persistence.mapper.impl;

import com.metawebthree.payment.domain.model.Rule;
import com.metawebthree.payment.enums.DecisionEnum;
import com.metawebthree.payment.enums.DeviceRiskTag;
import com.metawebthree.payment.infrastructure.persistence.mapper.RuleRepo;

import org.springframework.stereotype.Repository;

import java.util.ArrayList;
import java.util.List;

@Repository
public class RuleRepoImpl implements RuleRepo {

    @Override
    public List<Rule> load(String scene) {
        List<Rule> rules = new ArrayList<>();
        
        Rule r1 = new Rule();
        r1.setCode("D5RE");
        r1.setExpr(String.format("device_shared_degree > 5 && device_risk_tag != '%s'", DeviceRiskTag.NORMAL));
        r1.setAction(DecisionEnum.REJECT);
        r1.setPriority(10);
        rules.add(r1);

        Rule r2 = new Rule();
        r2.setCode("FYG3");
        r2.setExpr("first_order && gps_stability < 0.3");
        r2.setAction(DecisionEnum.REVIEW);
        r2.setPriority(100);
        rules.add(r2);

        return rules;
    }
}

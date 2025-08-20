package com.metawebthree.repository.impl;

import com.metawebthree.repository.RuleRepo;
import com.metawebthree.service.entity.Rule;

import org.springframework.stereotype.Repository;

import java.util.ArrayList;
import java.util.List;

@Repository
public class RuleRepoImpl implements RuleRepo {

    @Override
    public List<Rule> load(String scene) {
        List<Rule> rules = new ArrayList<>();
        
        Rule r1 = new Rule();
        r1.setCode("R203");
        r1.setExpr("device_shared_degree > 5 && device_risk_tag == 'emu'");
        r1.setAction("reject");
        r1.setPriority(10);
        rules.add(r1);

        Rule r2 = new Rule();
        r2.setCode("W101");
        r2.setExpr("first_order && gps_stability < 0.3");
        r2.setAction("review");
        r2.setPriority(100);
        rules.add(r2);

        return rules;
    }
}

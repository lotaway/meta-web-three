package com.metawebthree.payment.infrastructure.persistence.mapper;

import java.util.List;

import com.metawebthree.payment.domain.model.Rule;

public interface RuleRepo {
    List<Rule> load(String scene);
}

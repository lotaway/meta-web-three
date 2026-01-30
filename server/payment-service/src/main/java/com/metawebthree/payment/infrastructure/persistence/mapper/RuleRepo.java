package com.metawebthree.payment.infrastructure.persistence.mapper;
import com.metawebthree.payment.domain.model.*;

import java.util.List;

import com.metawebthree.entity.Rule;

public interface RuleRepo {
    List<Rule> load(String scene);
}

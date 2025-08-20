package com.metawebthree.service.repository;

import com.metawebthree.service.entity.Rule;
import java.util.List;

public interface RuleRepo {
    List<Rule> load(String scene);
}

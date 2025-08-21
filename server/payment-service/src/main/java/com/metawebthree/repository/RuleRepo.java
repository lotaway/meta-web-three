package com.metawebthree.repository;

import java.util.List;

import com.metawebthree.entity.Rule;

public interface RuleRepo {
    List<Rule> load(String scene);
}

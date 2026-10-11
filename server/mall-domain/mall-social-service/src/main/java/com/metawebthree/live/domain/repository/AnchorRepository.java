package com.metawebthree.live.domain.repository;

import com.metawebthree.live.domain.model.Anchor;

import java.util.List;

public interface AnchorRepository {
    Anchor save(Anchor anchor);
    Anchor findById(Long id);
    Anchor findByUserId(Long userId);
    List<Anchor> findAll();
    void deleteById(Long id);
}
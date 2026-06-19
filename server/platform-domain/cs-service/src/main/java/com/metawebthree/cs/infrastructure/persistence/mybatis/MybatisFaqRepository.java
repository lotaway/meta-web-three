package com.metawebthree.cs.infrastructure.persistence.mybatis;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.metawebthree.cs.domain.model.Faq;
import com.metawebthree.cs.domain.repository.FaqRepository;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
@RequiredArgsConstructor
public class MybatisFaqRepository implements FaqRepository {

    private final MybatisFaqMapper faqMapper;

    @Override
    public Faq save(Faq faq) {
        if (faq.getId() == null) {
            faqMapper.insert(faq);
        } else {
            faq.setUpdateTime(java.time.LocalDateTime.now());
            faqMapper.updateById(faq);
        }
        return faq;
    }

    @Override
    public Faq findById(Long id) {
        return faqMapper.selectById(id);
    }

    @Override
    public List<Faq> findAll() {
        return faqMapper.selectList(new LambdaQueryWrapper<Faq>()
                .orderByDesc(Faq::getPriority)
                .orderByDesc(Faq::getHitCount));
    }

    @Override
    public IPage<Faq> findAllPaged(Page<Faq> page) {
        return faqMapper.selectPage(page, new LambdaQueryWrapper<Faq>()
                .orderByDesc(Faq::getPriority)
                .orderByDesc(Faq::getHitCount));
    }

    @Override
    public List<Faq> findByCategory(String category) {
        return faqMapper.findByCategory(category);
    }

    @Override
    public IPage<Faq> findByCategoryPaged(Page<Faq> page, String category) {
        return faqMapper.selectPage(page, new LambdaQueryWrapper<Faq>()
                .eq(Faq::getCategory, category)
                .orderByDesc(Faq::getPriority)
                .orderByDesc(Faq::getHitCount));
    }

    @Override
    public List<Faq> findByEnabled(Boolean enabled) {
        return faqMapper.findByEnabled(enabled);
    }

    @Override
    public List<Faq> searchByKeyword(String keyword) {
        return faqMapper.searchByKeyword(keyword);
    }

    @Override
    public IPage<Faq> searchByKeywordPaged(Page<Faq> page, String keyword) {
        return faqMapper.selectPage(page, new LambdaQueryWrapper<Faq>()
                .like(Faq::getQuestion, keyword)
                .or().like(Faq::getAnswer, keyword)
                .or().like(Faq::getKeywords, keyword)
                .orderByDesc(Faq::getHitCount));
    }

    @Override
    public List<Faq> findTopByRelevance(int limit) {
        return faqMapper.findTopByRelevance(limit);
    }

    @Override
    public void deleteById(Long id) {
        faqMapper.deleteById(id);
    }

    @Override
    public Long count() {
        return faqMapper.selectCount(null);
    }
}
package com.metawebthree.cs.application;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.metawebthree.cs.domain.model.Faq;
import com.metawebthree.cs.domain.repository.FaqRepository;
import org.springframework.cache.annotation.CacheEvict;
import org.springframework.cache.annotation.Cacheable;
import org.springframework.stereotype.Service;

import java.util.Comparator;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

@Service
public class FaqService {
    private final FaqRepository faqRepository;

    public FaqService(FaqRepository faqRepository) {
        this.faqRepository = faqRepository;
    }

    @CacheEvict(value = "faq", allEntries = true)
    public Faq createFaq(String question, String answer, String category, List<String> keywords) {
        Faq faq = new Faq(question, answer, category, keywords);
        return faqRepository.save(faq);
    }

    @CacheEvict(value = "faq", allEntries = true)
    public Faq updateFaq(Long id, String question, String answer, String category, 
                        List<String> keywords, Boolean enabled, Integer priority) {
        Faq faq = faqRepository.findById(id);
        if (faq != null) {
            if (question != null) faq.setQuestion(question);
            if (answer != null) faq.setAnswer(answer);
            if (category != null) faq.setCategory(category);
            if (keywords != null) faq.setKeywords(keywords);
            if (enabled != null) faq.setEnabled(enabled);
            if (priority != null) faq.setPriority(priority);
            return faqRepository.save(faq);
        }
        return null;
    }

    @Cacheable(value = "faq", key = "#id", unless = "#result == null")
    public Optional<Faq> getFaq(Long id) {
        return Optional.ofNullable(faqRepository.findById(id));
    }

    @Cacheable(value = "faq", key = "'all'", unless = "#result == null || #result.getRecords().isEmpty()")
    public IPage<Faq> getAllFaqsPaged(int pageNum, int pageSize) {
        Page<Faq> page = new Page<>(pageNum, pageSize);
        return faqRepository.findAllPaged(page);
    }

    public List<Faq> getAllFaqs() {
        return faqRepository.findAll();
    }

    @Cacheable(value = "faq", key = "'category:' + #category", unless = "#result == null || #result.isEmpty()")
    public IPage<Faq> getFaqsByCategoryPaged(String category, int pageNum, int pageSize) {
        Page<Faq> page = new Page<>(pageNum, pageSize);
        return faqRepository.findByCategoryPaged(page, category);
    }

    public List<Faq> getFaqsByCategory(String category) {
        return faqRepository.findByCategory(category);
    }

    @CacheEvict(value = "faq", allEntries = true)
    public void deleteFaq(Long id) {
        faqRepository.deleteById(id);
    }

    @Cacheable(value = "faq", key = "'match:' + #query", unless = "#result == null")
    public Faq searchAndMatch(String query) {
        if (query == null || query.isEmpty()) {
            return null;
        }
        
        List<Faq> allFaqs = faqRepository.findByEnabled(true);
        
        List<Faq> matchedFaqs = allFaqs.stream()
            .filter(faq -> faq.matches(query))
            .collect(Collectors.toList());
        
        if (matchedFaqs.isEmpty()) {
            return null;
        }
        
        Faq bestMatch = matchedFaqs.stream()
            .max(Comparator.comparingInt(Faq::getHitCount))
            .orElse(null);
        
        if (bestMatch != null) {
            bestMatch.incrementHitCount();
            faqRepository.save(bestMatch);
        }
        
        return bestMatch;
    }

    public List<Faq> searchTopRelevance(int limit) {
        return faqRepository.findTopByRelevance(limit);
    }

    @Cacheable(value = "faq", key = "'keyword:' + #keyword", unless = "#result == null || #result.isEmpty()")
    public IPage<Faq> searchByKeywordPaged(String keyword, int pageNum, int pageSize) {
        Page<Faq> page = new Page<>(pageNum, pageSize);
        return faqRepository.searchByKeywordPaged(page, keyword);
    }

    public List<Faq> searchByKeyword(String keyword) {
        return faqRepository.searchByKeyword(keyword);
    }

    public Faq autoReply(String userMessage) {
        Faq matchedFaq = searchAndMatch(userMessage);
        
        if (matchedFaq != null) {
            return matchedFaq;
        }
        
        return null;
    }

    @Cacheable(value = "faq", key = "'count'")
    public Long getTotalCount() {
        return faqRepository.count();
    }
}
package com.metawebthree.cs.application;

import com.metawebthree.cs.domain.model.Faq;
import com.metawebthree.cs.domain.repository.FaqRepository;

import java.util.Comparator;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

public class FaqService {
    private final FaqRepository faqRepository;

    public FaqService(FaqRepository faqRepository) {
        this.faqRepository = faqRepository;
    }

    public Faq createFaq(String question, String answer, String category, List<String> keywords) {
        Faq faq = new Faq(question, answer, category, keywords);
        return faqRepository.save(faq);
    }

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

    public Optional<Faq> getFaq(Long id) {
        return Optional.ofNullable(faqRepository.findById(id));
    }

    public List<Faq> getAllFaqs() {
        return faqRepository.findAll();
    }

    public List<Faq> getFaqsByCategory(String category) {
        return faqRepository.findByCategory(category);
    }

    public void deleteFaq(Long id) {
        faqRepository.deleteById(id);
    }

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

    public Long getTotalCount() {
        return faqRepository.count();
    }
}
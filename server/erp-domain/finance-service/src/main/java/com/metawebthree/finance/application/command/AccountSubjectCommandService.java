package com.metawebthree.finance.application.command;

import com.metawebthree.finance.domain.entity.AccountSubject;
import com.metawebthree.finance.domain.repository.AccountSubjectRepository;
import org.springframework.stereotype.Service;

@Service
public class AccountSubjectCommandService {
    private final AccountSubjectRepository subjectRepository;

    public AccountSubjectCommandService(AccountSubjectRepository subjectRepository) {
        this.subjectRepository = subjectRepository;
    }

    public Long createSubject(String subjectCode, String subjectName, String direction, Long parentId) {
        AccountSubject.SubjectDirection dir = AccountSubject.SubjectDirection.valueOf(direction.toUpperCase());
        AccountSubject subject = new AccountSubject();
        subject.create(subjectCode, subjectName, dir, parentId);
        subjectRepository.save(subject);
        return subject.getId();
    }

    public void disable(Long subjectId) {
        AccountSubject subject = subjectRepository.findById(subjectId)
            .orElseThrow(() -> new IllegalArgumentException("Subject not found"));
        subject.disable();
        subjectRepository.update(subject);
    }

    public void enable(Long subjectId) {
        AccountSubject subject = subjectRepository.findById(subjectId)
            .orElseThrow(() -> new IllegalArgumentException("Subject not found"));
        subject.enable();
        subjectRepository.update(subject);
    }

    public void updateBalance(Long subjectId, java.math.BigDecimal amount) {
        AccountSubject subject = subjectRepository.findById(subjectId)
            .orElseThrow(() -> new IllegalArgumentException("Subject not found"));
        subject.updateBalance(amount);
        subjectRepository.update(subject);
    }
}
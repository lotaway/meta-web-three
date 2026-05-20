package com.metawebthree.finance.application.query;

import com.metawebthree.finance.domain.entity.AccountSubject;
import com.metawebthree.finance.domain.repository.AccountSubjectRepository;
import org.springframework.stereotype.Service;
import java.util.List;
import java.util.Optional;

@Service
public class AccountSubjectQueryService {
    private final AccountSubjectRepository subjectRepository;

    public AccountSubjectQueryService(AccountSubjectRepository subjectRepository) {
        this.subjectRepository = subjectRepository;
    }

    public Optional<AccountSubject> getById(Long subjectId) {
        return subjectRepository.findById(subjectId);
    }

    public Optional<AccountSubject> getBySubjectCode(String subjectCode) {
        return subjectRepository.findBySubjectCode(subjectCode);
    }

    public List<AccountSubject> listActiveSubjects() {
        return subjectRepository.findByStatus(AccountSubject.SubjectStatus.ACTIVE);
    }

    public List<AccountSubject> listAll() {
        return subjectRepository.findAll();
    }

    public List<AccountSubject> listByLevel(Integer level) {
        return subjectRepository.findByLevel(level);
    }

    public List<AccountSubject> listByParentId(Long parentId) {
        return subjectRepository.findByParentId(parentId);
    }
}
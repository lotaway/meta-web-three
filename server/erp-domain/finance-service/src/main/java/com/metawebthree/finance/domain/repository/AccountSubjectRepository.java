package com.metawebthree.finance.domain.repository;

import com.metawebthree.finance.domain.entity.AccountSubject;
import java.util.List;
import java.util.Optional;

public interface AccountSubjectRepository {
    Optional<AccountSubject> findById(Long id);
    Optional<AccountSubject> findBySubjectCode(String subjectCode);
    List<AccountSubject> findBySubjectCodeLike(String subjectCodePrefix);
    List<AccountSubject> findByParentId(Long parentId);
    List<AccountSubject> findByLevel(Integer level);
    List<AccountSubject> findByStatus(AccountSubject.SubjectStatus status);
    List<AccountSubject> findAll();
    void save(AccountSubject subject);
    void update(AccountSubject subject);
    void delete(Long id);
}
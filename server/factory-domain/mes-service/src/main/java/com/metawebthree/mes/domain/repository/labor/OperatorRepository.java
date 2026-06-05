package com.metawebthree.mes.domain.repository.labor;

import com.metawebthree.mes.domain.entity.labor.Operator;
import java.util.List;
import java.util.Optional;

public interface OperatorRepository {
    Optional<Operator> findById(Long id);
    Optional<Operator> findByOperatorCode(String operatorCode);
    List<Operator> findByDepartment(String department);
    List<Operator> findByStatus(Operator.OperatorStatus status);
    List<Operator> findByShiftGroup(String shiftGroup);
    List<Operator> findAll();
    Operator save(Operator operator);
    void update(Operator operator);
    void deleteById(Long id);
}

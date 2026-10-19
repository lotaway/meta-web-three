package com.metawebthree.rma.domain.repository;

import com.metawebthree.rma.domain.entity.RmaDisposition;
import com.metawebthree.rma.infrastructure.persistence.dataobject.RmaDispositionDO;
import org.apache.ibatis.annotations.Param;
import org.springframework.stereotype.Repository;
import java.util.Optional;

@Repository
public interface RmaDispositionRepository {
    Optional<RmaDisposition> findByRmaId(Long rmaId);
    RmaDisposition save(RmaDisposition disposition);

    @Repository
    interface RmaDispositionRepositoryImpl {
        default Optional<RmaDisposition> findByIdempotentKey(String idempotentKey) {
            // 通过 MyBatis-Plus 查询，实际映射由 XML 或注解实现
            // 此处为默认实现，子类或 XML 需要提供具体查询语句
            return null;
        }
    }
}

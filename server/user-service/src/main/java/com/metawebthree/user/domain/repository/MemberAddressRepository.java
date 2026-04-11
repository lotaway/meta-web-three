package com.metawebthree.user.domain.repository;

import com.metawebthree.user.domain.model.MemberAddress;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Modifying;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;
import java.util.List;
import java.util.Optional;

@Repository
public interface MemberAddressRepository extends JpaRepository<MemberAddress, Long> {

    List<MemberAddress> findByMemberId(Long memberId);

    Optional<MemberAddress> findById(Long id);

    @Modifying
    @Query("UPDATE MemberAddress a SET a.defaultStatus = false WHERE a.memberId = :memberId")
    void clearDefaultStatus(@Param("memberId") Long memberId);

    // save, update, delete 由 JpaRepository 提供
}

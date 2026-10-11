package com.metawebthree.user.domain.repository;

import com.metawebthree.user.domain.model.MemberAddress;
import java.util.List;
import java.util.Optional;

public interface MemberAddressRepository {

    void save(MemberAddress address);

    void update(MemberAddress address);

    List<MemberAddress> findByMemberId(Long memberId);

    Optional<MemberAddress> findById(Long id);

    void deleteById(Long id);

    void clearDefaultStatus(Long memberId);
}

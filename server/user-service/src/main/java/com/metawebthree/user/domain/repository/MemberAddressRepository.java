package com.metawebthree.user.domain.repository;

import com.metawebthree.user.domain.model.MemberAddress;
import java.util.List;

public interface MemberAddressRepository {
    void save(MemberAddress address);
    void update(MemberAddress address);
    MemberAddress findById(Long id);
    List<MemberAddress> findByMemberId(Long memberId);
    void delete(Long id);
    void clearDefaultStatus(Long memberId);
}

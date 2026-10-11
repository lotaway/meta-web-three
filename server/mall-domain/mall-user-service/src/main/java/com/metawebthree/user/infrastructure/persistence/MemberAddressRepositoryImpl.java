package com.metawebthree.user.infrastructure.persistence;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.conditions.update.LambdaUpdateWrapper;
import com.metawebthree.user.domain.model.MemberAddress;
import com.metawebthree.user.domain.repository.MemberAddressRepository;
import com.metawebthree.user.infrastructure.persistence.mapper.MemberAddressMapper;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
@RequiredArgsConstructor
public class MemberAddressRepositoryImpl implements MemberAddressRepository {

    private final MemberAddressMapper memberAddressMapper;

    @Override
    public void save(MemberAddress address) {
        if (address.getId() == null) {
            memberAddressMapper.insert(address);
        } else {
            memberAddressMapper.updateById(address);
        }
    }

    @Override
    public void update(MemberAddress address) {
        memberAddressMapper.updateById(address);
    }

    @Override
    public Optional<MemberAddress> findById(Long id) {
        return Optional.ofNullable(memberAddressMapper.selectById(id));
    }

    @Override
    public List<MemberAddress> findByMemberId(Long memberId) {
        return memberAddressMapper.selectList(new LambdaQueryWrapper<MemberAddress>()
                .eq(MemberAddress::getMemberId, memberId));
    }

    @Override
    public void deleteById(Long id) {
        memberAddressMapper.deleteById(id);
    }

    @Override
    public void clearDefaultStatus(Long memberId) {
        memberAddressMapper.update(null, new LambdaUpdateWrapper<MemberAddress>()
                .set(MemberAddress::isDefaultStatus, false)
                .eq(MemberAddress::getMemberId, memberId)
                .eq(MemberAddress::isDefaultStatus, true));
    }
}

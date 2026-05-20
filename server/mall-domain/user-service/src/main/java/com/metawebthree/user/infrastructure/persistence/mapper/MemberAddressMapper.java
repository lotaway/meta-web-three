package com.metawebthree.user.infrastructure.persistence.mapper;
import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.user.domain.model.MemberAddress;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface MemberAddressMapper extends BaseMapper<MemberAddress> {
}

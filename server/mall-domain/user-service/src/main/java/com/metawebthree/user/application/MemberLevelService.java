package com.metawebthree.user.application;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.metawebthree.user.domain.model.MemberLevelDO;
import com.metawebthree.user.infrastructure.persistence.mapper.MemberLevelMapper;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
@RequiredArgsConstructor
public class MemberLevelService extends ServiceImpl<MemberLevelMapper, MemberLevelDO> {

    private final MemberLevelMapper memberLevelMapper;

    public List<MemberLevelDO> listByDefaultStatus(Integer defaultStatus) {
        LambdaQueryWrapper<MemberLevelDO> wrapper = new LambdaQueryWrapper<>();
        if (defaultStatus != null) {
            wrapper.eq(MemberLevelDO::getDefaultStatus, defaultStatus);
        }
        return memberLevelMapper.selectList(wrapper);
    }
}

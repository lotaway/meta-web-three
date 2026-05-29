package com.metawebthree.finance.infrastructure.persistence.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.finance.infrastructure.persistence.dataobject.VoucherLineDO;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Select;
import java.util.List;

@Mapper
public interface VoucherLineMapper extends BaseMapper<VoucherLineDO> {
    
    @Select("SELECT * FROM finance_voucher_line WHERE voucher_id = #{voucherId}")
    List<VoucherLineDO> selectByVoucherId(Long voucherId);
}
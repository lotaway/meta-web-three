package com.metawebthree.procurement.infrastructure.persistence.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.procurement.infrastructure.persistence.dataobject.ProcurementReturnOrderDO;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface ProcurementReturnOrderMapper extends BaseMapper<ProcurementReturnOrderDO> {
}
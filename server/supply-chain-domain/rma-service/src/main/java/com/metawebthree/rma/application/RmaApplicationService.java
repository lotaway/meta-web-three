package com.metawebthree.rma.application;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.metawebthree.rma.application.dto.*;
import java.util.List;

public interface RmaApplicationService {

    RmaOrderDTO createRma(CreateRmaRequest request);

    RmaOrderDTO getRma(Long id);

    RmaOrderDTO getRmaByNo(String rmaNo);

    IPage<RmaOrderDTO> listRmas(String status, Integer pageNum, Integer pageSize);

    RmaOrderDTO submitForInspection(Long rmaId);

    RmaOrderDTO recordInspection(Long rmaId, RecordInspectionRequest request);

    RmaOrderDTO makeDisposition(Long rmaId, MakeDispositionRequest request);

    RmaOrderDTO executeDisposition(Long rmaId);

    RmaOrderDTO completeRma(Long rmaId);

    RmaOrderDTO cancelRma(Long rmaId);

    List<?> getRmaTimeline(Long rmaId);
}

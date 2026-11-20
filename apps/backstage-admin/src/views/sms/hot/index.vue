<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Search, Tickets } from '@element-plus/icons-vue'
import { getHomeRecommendProductListAPI, homeRecommendProductUpdateRecommendStatusAPI, homeRecommendProductDeleteByIdsAPI, homeRecommendProductCreateAPI, homeRecommendProductUpdateSortByIdAPI } from '@/apis/recommendProduct'
import { getProductListAPI } from '@/apis/product'
import type { RecommendProductQueryParam, SmsHomeRecommendProduct } from '@/types/recommendProduct'
import type { PmsProduct } from '@/types/product'
import { useI18n } from 'vue-i18n'

const { t } = useI18n()

// List query params
const listQuery = ref<RecommendProductQueryParam>({
  pageNum: 1,
  pageSize: 10
})
// Selected state in query conditions
const recommendOptions = ref([
  {
    label: '未推荐',
    value: 0
  },
  {
    label: '推荐中',
    value: 1
  }
])
// New-product list data
const list = ref<SmsHomeRecommendProduct[]>([])
// Total count
const total = ref(0)
// Loading state
const listLoading = ref(false)
// Get list data
const getList = async () => {
  listLoading.value = true;
  const res = await getHomeRecommendProductListAPI(listQuery.value)
  listLoading.value = false;
  list.value = res.data.list;
  total.value = res.data.total;
};

// Run after component mounts
onMounted(() => {
  getList()
})

// Data in the select-product dialog
const dialogData = ref({
  list: [] as PmsProduct[],
  total: 0,
  multipleSelection: [] as PmsProduct[],
  listQuery: {
    keyword: '',
    pageNum: 1,
    pageSize: 5
  }
})
// Select-product dialog visibility
const selectDialogVisible = ref(false)

// Batch operation selected records
const multipleSelection = ref<SmsHomeRecommendProduct[]>([])
// Batch operation enum
const operates = ref([
  {
    label: "设为推荐",
    value: 0
  },
  {
    label: "取消推荐",
    value: 1
  },
  {
    label: "删除",
    value: 2
  }
])
// Batch operation type
const operateType = ref()

// Set sort dialog visibility
const sortDialogVisible = ref(false)
// Set sort dialog data
const sortDialogData = ref({ sort: 0, id: 0 })

// Reset search
const handleResetSearch = () => {
  listQuery.value = { pageNum: 1, pageSize: 10 }
};

// Search list
const handleSearchList = () => {
  listQuery.value.pageNum = 1
  getList()
}

// Handle table selection change
const handleSelectionChange = (val: SmsHomeRecommendProduct[]) => {
  multipleSelection.value = val
}

// Handle page size change
const handleSizeChange = (val: number) => {
  listQuery.value.pageNum = 1
  listQuery.value.pageSize = val
  getList()
}

// Handle current page change
const handleCurrentChange = (val: number) => {
  listQuery.value.pageNum = val
  getList()
}

// Handle recommendation status change
const handleRecommendStatusStatusChange = (index: number, row: SmsHomeRecommendProduct) => {
  updateRecommendStatusStatus([row.id!], row.recommendStatus!)
}

// Handle delete
const handleDelete = async (index: number, row: SmsHomeRecommendProduct) => {
  deleteNewProduct([row.id!])
}

// Handle batch operations
const handleBatchOperate = async () => {
  if (multipleSelection.value.length < 1) {
    ElMessage.warning({
      message: '请选择一条记录',
      duration: 1000
    })
    return
  }
  const ids = multipleSelection.value.map(item => item.id!)
  if (operateType.value === 0) {
    // Set as recommended
    await updateRecommendStatusStatus(ids, 1)
  } else if (operateType.value === 1) {
    // Unset recommendation
    await updateRecommendStatusStatus(ids, 0)
  } else if (operateType.value === 2) {
    // Delete
    await deleteNewProduct(ids)
  } else {
    ElMessage.warning({
      message: '请选择批量操作类型',
      duration: 1000
    })
  }
}

// Handle product selection
const handleSelectProduct = () => {
  selectDialogVisible.value = true;
  getDialogList()
}

// Handle selection search
const handleSelectSearch = () => {
  getDialogList();
};

// Handle dialog page size change
const handleDialogSizeChange = (val: number) => {
  dialogData.value.listQuery.pageNum = 1
  dialogData.value.listQuery.pageSize = val
  getDialogList()
};

// Handle dialog current page change
const handleDialogCurrentChange = (val: number) => {
  dialogData.value.listQuery.pageNum = val
  getDialogList()
};

// Handle dialog selection change
const handleDialogSelectionChange = (val: PmsProduct[]) => {
  dialogData.value.multipleSelection = val
}

// Handle selection dialog confirmation
const handleSelectDialogConfirm = async () => {
  if (dialogData.value.multipleSelection.length < 1) {
    ElMessage.warning({
      message: '请选择一条记录',
      duration: 1000
    })
    return
  }
  await ElMessageBox.confirm(t('dialog.confirmAdd'), t('dialog.title'), {
    confirmButtonText: t('common.confirm'),
    cancelButtonText: t('common.cancel'),
    type: 'warning'
  })
  const selectProducts = dialogData.value.multipleSelection.map(item => ({ productId: item.id!, productName: item.name }))
  await homeRecommendProductCreateAPI(selectProducts)
  selectDialogVisible.value = false
  dialogData.value.multipleSelection = []
  getList()
  ElMessage.success({
    type: 'success',
    message: '添加成功!'
  })
}

// Handle edit sort
const handleEditSort = (index: number, row: SmsHomeRecommendProduct) => {
  sortDialogVisible.value = true
  sortDialogData.value.sort = row.sort!
  sortDialogData.value.id = row.id!
};

// Handle update sort
const handleUpdateSort = async () => {
  try {
    await ElMessageBox.confirm(t('dialog.confirmChangeSort'), t('dialog.title'), {
      confirmButtonText: t('common.confirm'),
      cancelButtonText: t('common.cancel'),
      type: 'warning'
    })
    await homeRecommendProductUpdateSortByIdAPI(sortDialogData.value);
    sortDialogVisible.value = false
    getList()
    ElMessage.success({
      type: 'success',
      message: '修改成功!'
    })
  } catch {
    ElMessage.info(t('message.cancelled'))
  }
}

// Update recommendation status
const updateRecommendStatusStatus = async (ids: number[], status: number) => {
  try {
    await ElMessageBox.confirm(t('dialog.confirmChangeRecommend'), t('dialog.title'), {
      confirmButtonText: t('common.confirm'),
      cancelButtonText: t('common.cancel'),
      type: 'warning'
    })
    await homeRecommendProductUpdateRecommendStatusAPI({ ids: ids.join(','), recommendStatus: status })
    getList()
    ElMessage.success({
      type: 'success',
      message: '修改成功!'
    })
  } catch {
    getList()
  }
}

// Delete product
const deleteNewProduct = async (ids: number[]) => {
  try {
    await ElMessageBox.confirm(t('dialog.confirmDeleteRecommend'), t('dialog.title'), {
      confirmButtonText: t('common.confirm'),
      cancelButtonText: t('common.cancel'),
      type: 'warning'
    })
    await homeRecommendProductDeleteByIdsAPI({ ids: ids.join(',') });
    getList()
    ElMessage.success({
      type: 'success',
      message: '删除成功!'
    })
  } catch {
    ElMessage.info(t('message.cancelled'))
  }
}

// Get dialog list
const getDialogList = async () => {
  const res = await getProductListAPI(dialogData.value.listQuery)
  dialogData.value.list = res.data.list;
  dialogData.value.total = res.data.total;
};

// Format recommendation status
const formatRecommendStatus = (status: number) => {
  if (status === 1) {
    return '推荐中';
  } else {
    return '未推荐';
  }
};
</script>

<template>
  <div class="app-container">
    <el-card class="filter-container" shadow="never">
      <div>
        <el-icon class="el-icon-middle">
          <Search />
        </el-icon>
        <span>筛选搜索</span>
        <el-button style="float:right" type="primary" @click="handleSearchList()">
          查询搜索
        </el-button>
        <el-button style="float:right;margin-right: 15px" @click="handleResetSearch()">
          重置
        </el-button>
      </div>
      <div style="margin-top: 15px">
        <el-form :inline="true" :model="listQuery" label-width="140px">
          <el-form-item label="商品名称：">
            <el-input v-model="listQuery.productName" class="input-width" placeholder="商品名称"></el-input>
          </el-form-item>
          <el-form-item label="推荐状态：">
            <el-select v-model="listQuery.recommendStatus" placeholder="全部" clearable class="input-width"
              style="width: 140px;">
              <el-option v-for="item in recommendOptions" :key="item.value" :label="item.label" :value="item.value">
              </el-option>
            </el-select>
          </el-form-item>
        </el-form>
      </div>
    </el-card>
    <el-card class="operate-container" shadow="never">
      <el-icon class="el-icon-middle">
        <Tickets />
      </el-icon>
      <span>数据列表</span>
      <el-button class="btn-add" @click="handleSelectProduct()">选择商品</el-button>
    </el-card>
    <div class="table-container">
      <el-table ref="newProductTable" :data="list" style="width: 100%;" @selection-change="handleSelectionChange"
        v-loading="listLoading" border>
        <el-table-column type="selection" width="60" align="center"></el-table-column>
        <el-table-column label="编号" width="120" align="center">
          <template #default="scope">{{ scope.row.id }}</template>
        </el-table-column>
        <el-table-column label="商品名称" align="center">
          <template #default="scope">{{ scope.row.productName }}</template>
        </el-table-column>
        <el-table-column label="是否推荐" width="200" align="center">
          <template #default="scope">
            <el-switch @change="handleRecommendStatusStatusChange(scope.$index, scope.row as SmsHomeRecommendProduct)" :active-value="1"
              :inactive-value="0" v-model="scope.row.recommendStatus">
            </el-switch>
          </template>
        </el-table-column>
        <el-table-column label="排序" width="160" align="center">
          <template #default="scope">{{ scope.row.sort }}</template>
        </el-table-column>
        <el-table-column label="状态" width="160" align="center">
          <template #default="scope">{{ formatRecommendStatus(scope.row.recommendStatus) }}</template>
        </el-table-column>
        <el-table-column label="操作" width="180" align="center">
          <template #default="scope">
            <el-button size="small" type="primary" link @click="handleEditSort(scope.$index, scope.row as SmsHomeRecommendProduct)">设置排序
            </el-button>
            <el-button size="small" type="primary" link @click="handleDelete(scope.$index, scope.row as SmsHomeRecommendProduct)">删除
            </el-button>
          </template>
        </el-table-column>
      </el-table>
    </div>
    <div class="batch-operate-container">
      <el-select v-model="operateType" placeholder="批量操作">
        <el-option v-for="item in operates" :key="item.value" :label="item.label" :value="item.value">
        </el-option>
      </el-select>
      <el-button style="margin-left: 20px" class="search-button" @click="handleBatchOperate()" type="primary">
        确定
      </el-button>
    </div>
    <div class="pagination-container">
      <el-pagination background @size-change="handleSizeChange" @current-change="handleCurrentChange"
        layout="total, sizes,prev, pager, next,jumper" :page-size="listQuery.pageSize" :page-sizes="[5, 10, 15]"
        v-model:current-page="listQuery.pageNum" :total="total">
      </el-pagination>
    </div>
    <el-dialog v-model="selectDialogVisible" title="选择商品" width="50%">
      <el-input v-model="dialogData.listQuery.keyword" style="width: 250px;margin-bottom: 20px" placeholder="商品名称搜索">
        <template #append>
          <el-button :icon="Search" @click="handleSelectSearch()"></el-button>
        </template>
      </el-input>
      <el-table :data="dialogData.list" @selection-change="handleDialogSelectionChange" border>
        <el-table-column type="selection" width="60" align="center"></el-table-column>
        <el-table-column label="商品名称" align="center">
          <template #default="scope">{{ scope.row.name }}</template>
        </el-table-column>
        <el-table-column label="货号" width="160" align="center">
          <template #default="scope">NO.{{ scope.row.productSn }}</template>
        </el-table-column>
        <el-table-column label="价格" width="120" align="center">
          <template #default="scope">￥{{ scope.row.price }}</template>
        </el-table-column>
      </el-table>
      <div class="pagination-container">
        <el-pagination background @size-change="handleDialogSizeChange" @current-change="handleDialogCurrentChange"
          layout="prev, pager, next" v-model:current-page="dialogData.listQuery.pageNum"
          :page-size="dialogData.listQuery.pageSize" :page-sizes="[5, 10, 15]" :total="dialogData.total">
        </el-pagination>
      </div>
      <div style="clear: both;"></div>
      <template #footer>
        <div>
          <el-button @click="selectDialogVisible = false">取 消</el-button>
          <el-button type="primary" @click="handleSelectDialogConfirm()">确 定</el-button>
        </div>
      </template>
    </el-dialog>
    <el-dialog v-model="sortDialogVisible" title="设置排序" width="40%">
      <el-form :model="sortDialogData" label-width="150px">
        <el-form-item label="排序：">
          <el-input v-model="sortDialogData.sort" style="width: 200px"></el-input>
        </el-form-item>
      </el-form>
      <template #footer>
        <span>
          <el-button @click="sortDialogVisible = false" size="small">取 消</el-button>
          <el-button type="primary" @click="handleUpdateSort" size="small">确 定</el-button>
        </span>
      </template>
    </el-dialog>
  </div>
</template>

<style></style>

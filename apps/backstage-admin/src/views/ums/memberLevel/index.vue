<template>
  <div class="member-level-container">
    <el-card class="filter-card">
      <el-form :inline="true" :model="listQuery">
        <el-form-item label="等级名称">
          <el-input v-model="listQuery.name" placeholder="请输入等级名称" clearable @keyup.enter="handleSearch" />
        </el-form-item>
        <el-form-item label="默认等级">
          <el-select v-model="listQuery.defaultStatus" placeholder="请选择" clearable>
            <el-option label="是" :value="1" />
            <el-option label="否" :value="0" />
          </el-select>
        </el-form-item>
        <el-form-item>
          <el-button type="primary" @click="handleSearch">查询</el-button>
          <el-button @click="handleReset">重置</el-button>
        </el-form-item>
      </el-form>
    </el-card>

    <el-card class="table-card">
      <div class="toolbar">
        <el-button type="primary" @click="handleAdd">添加会员等级</el-button>
      </div>

      <el-table v-loading="listLoading" :data="list" border stripe>
        <el-table-column prop="id" label="ID" width="80" />
        <el-table-column prop="name" label="等级名称" width="120" />
        <el-table-column prop="growthPoint" label="成长值要求" width="120" />
        <el-table-column prop="defaultStatus" label="默认等级" width="100">
          <template #default="{ row }">
            <el-tag :type="row.defaultStatus === 1 ? 'success' : 'info'">
              {{ row.defaultStatus === 1 ? '是' : '否' }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="freeFreightPoint" label="免运费标准" width="120" />
        <el-table-column prop="commentGrowthPoint" label="评价成长值" width="120" />
        <el-table-column label="特权" min-width="200">
          <template #default="{ row }">
            <el-tag v-if="row.priviledgeFreeFreight" size="small" type="success">免邮</el-tag>
            <el-tag v-if="row.priviledgeSignIn" size="small" type="success">签到</el-tag>
            <el-tag v-if="row.priviledgeComment" size="small" type="success">评论</el-tag>
            <el-tag v-if="row.priviledgePromotion" size="small" type="success">活动</el-tag>
            <el-tag v-if="row.priviledgeMemberPrice" size="small" type="success">会员价</el-tag>
            <el-tag v-if="row.priviledgeBirthday" size="small" type="success">生日</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="note" label="备注" min-width="150" show-overflow-tooltip />
        <el-table-column label="操作" width="180" fixed="right">
          <template #default="{ row }">
            <el-button type="primary" link @click="handleEdit(row)">编辑</el-button>
            <el-button type="danger" link @click="handleDelete(row)">删除</el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <!-- 添加/编辑对话框 -->
    <el-dialog v-model="dialogVisible" :title="isEdit ? '编辑会员等级' : '添加会员等级'" width="600px">
      <el-form ref="formRef" :model="form" :rules="rules" label-width="120px">
        <el-form-item label="等级名称" prop="name">
          <el-input v-model="form.name" placeholder="请输入等级名称" />
        </el-form-item>
        <el-form-item label="成长值要求" prop="growthPoint">
          <el-input-number v-model="form.growthPoint" :min="0" :max="999999" />
        </el-form-item>
        <el-form-item label="默认等级" prop="defaultStatus">
          <el-radio-group v-model="form.defaultStatus">
            <el-radio :value="1">是</el-radio>
            <el-radio :value="0">否</el-radio>
          </el-radio-group>
        </el-form-item>
        <el-form-item label="免运费标准" prop="freeFreightPoint">
          <el-input-number v-model="form.freeFreightPoint" :min="0" :precision="2" />
        </el-form-item>
        <el-form-item label="评价成长值" prop="commentGrowthPoint">
          <el-input-number v-model="form.commentGrowthPoint" :min="0" />
        </el-form-item>
        <el-divider>特权设置</el-divider>
        <el-form-item label="免邮特权">
          <el-switch v-model="form.priviledgeFreeFreight" :active-value="1" :inactive-value="0" />
        </el-form-item>
        <el-form-item label="签到特权">
          <el-switch v-model="form.priviledgeSignIn" :active-value="1" :inactive-value="0" />
        </el-form-item>
        <el-form-item label="评论特权">
          <el-switch v-model="form.priviledgeComment" :active-value="1" :inactive-value="0" />
        </el-form-item>
        <el-form-item label="专享活动特权">
          <el-switch v-model="form.priviledgePromotion" :active-value="1" :inactive-value="0" />
        </el-form-item>
        <el-form-item label="会员价特权">
          <el-switch v-model="form.priviledgeMemberPrice" :active-value="1" :inactive-value="0" />
        </el-form-item>
        <el-form-item label="生日特权">
          <el-switch v-model="form.priviledgeBirthday" :active-value="1" :inactive-value="0" />
        </el-form-item>
        <el-form-item label="备注">
          <el-input v-model="form.note" type="textarea" :rows="3" placeholder="请输入备注" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="dialogVisible = false">取消</el-button>
        <el-button type="primary" @click="handleSubmit">确定</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, onMounted } from 'vue'
import { ElMessage, ElMessageBox, type FormInstance } from 'element-plus'
import {
  getMemberLevelListAPI,
  createMemberLevelAPI,
  updateMemberLevelAPI,
  deleteMemberLevelAPI,
} from '@/apis/memberLevel'
import type { UmsMemberLevel } from '@/types/memberLevel'

// 列表查询参数
const listQuery = ref<{
  name: string
  defaultStatus: number | undefined
}>({
  name: '',
  defaultStatus: undefined,
})

// 列表数据
const list = ref<UmsMemberLevel[]>([])
const listLoading = ref(false)

// 对话框
const dialogVisible = ref(false)
const isEdit = ref(false)
const formRef = ref<FormInstance>()

// 表单数据
const form = reactive<UmsMemberLevel>({
  name: '',
  growthPoint: 0,
  defaultStatus: 0,
  freeFreightPoint: 0,
  commentGrowthPoint: 0,
  priviledgeFreeFreight: 0,
  priviledgeSignIn: 0,
  priviledgeComment: 0,
  priviledgePromotion: 0,
  priviledgeMemberPrice: 0,
  priviledgeBirthday: 0,
  note: '',
})

// 表单验证规则
const rules = {
  name: [{ required: true, message: '请输入等级名称', trigger: 'blur' }],
  growthPoint: [{ required: true, message: '请输入成长值要求', trigger: 'blur' }],
}

// 获取列表
const getList = async () => {
  listLoading.value = true
  try {
    const res = await getMemberLevelListAPI({
      defaultStatus: listQuery.value.defaultStatus,
    })
    let data = res.data || []
    // 根据名称过滤
    if (listQuery.value.name) {
      data = data.filter((item) => item.name.includes(listQuery.value.name))
    }
    list.value = data
  } catch (error) {
    console.error('获取会员等级列表失败:', error)
  } finally {
    listLoading.value = false
  }
}

// 搜索
const handleSearch = () => {
  getList()
}

// 重置
const handleReset = () => {
  listQuery.value = {
    name: '',
    defaultStatus: undefined,
  }
  getList()
}

// 添加
const handleAdd = () => {
  isEdit.value = false
  Object.assign(form, {
    id: undefined,
    name: '',
    growthPoint: 0,
    defaultStatus: 0,
    freeFreightPoint: 0,
    commentGrowthPoint: 0,
    priviledgeFreeFreight: 0,
    priviledgeSignIn: 0,
    priviledgeComment: 0,
    priviledgePromotion: 0,
    priviledgeMemberPrice: 0,
    priviledgeBirthday: 0,
    note: '',
  })
  dialogVisible.value = true
}

// 编辑
const handleEdit = (row: UmsMemberLevel) => {
  isEdit.value = true
  Object.assign(form, row)
  dialogVisible.value = true
}

// 提交表单
const handleSubmit = async () => {
  await formRef.value?.validate()
  try {
    if (isEdit.value && form.id) {
      await updateMemberLevelAPI(form.id, form)
      ElMessage.success('更新成功')
    } else {
      await createMemberLevelAPI(form)
      ElMessage.success('创建成功')
    }
    dialogVisible.value = false
    getList()
  } catch (error) {
    console.error('提交失败:', error)
    ElMessage.error('操作失败')
  }
}

// 删除
const handleDelete = (row: UmsMemberLevel) => {
  ElMessageBox.confirm('确定要删除该会员等级吗?', '提示', {
    confirmButtonText: '确定',
    cancelButtonText: '取消',
    type: 'warning',
  }).then(async () => {
    try {
      await deleteMemberLevelAPI(row.id!)
      ElMessage.success('删除成功')
      getList()
    } catch (error) {
      console.error('删除失败:', error)
      ElMessage.error('删除失败')
    }
  })
}

onMounted(() => {
  getList()
})
</script>

<style scoped>
.member-level-container {
  padding: 20px;
}

.filter-card {
  margin-bottom: 20px;
}

.table-card .toolbar {
  margin-bottom: 16px;
}
</style>
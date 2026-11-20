<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { dayjs, ElMessage, ElMessageBox } from 'element-plus'
import { Search, Tickets } from '@element-plus/icons-vue'
import { getAdminListAPI, adminRegisterAPI, adminUpdateByIdAPI, adminUpdateStatusByIdAPI, adminDeleteByIdAPI, getRoleByAdminIdAPI, adminRoleUpdateAPI } from '@/apis/admin.ts'
import { getRoleListAllAPI } from '@/apis/role.ts'
import type { UmsAdmin } from '@/types/admin'
import type { PageParam } from '@/types/common'
import type { UmsRole } from '@/types/role'

// List query params
const listQuery = ref<PageParam>({
  pageNum: 1,
  pageSize: 10,
  keyword: ''
})
// Admin list data
const list = ref<UmsAdmin[]>([])
// All role list
const allRoleList = ref<UmsRole[]>([])
// Table loading state
const listLoading = ref(true)
// Pagination total
const total = ref(0)
// Get admin list
const getList = async () => {
  listLoading.value = true
  try {
    const res = await getAdminListAPI(listQuery.value)
    listLoading.value = false
    list.value = res.data.list
    total.value = res.data.total
  } catch (error) {
    listLoading.value = false
    console.error('获取管理员列表失败:', error)
  }
}
// Get all roles
const getAllRoleList = async () => {
  try {
    const response = await getRoleListAllAPI()
    allRoleList.value = response.data
  } catch (error) {
    console.error('获取角色列表失败:', error)
  }
}

// Initialize data after mount
onMounted(() => {
  getList()
  getAllRoleList()
})

// Current admin
const admin = ref<UmsAdmin>({
  username: '',
  password: '',
  status: 1
})
// Whether the admin edit dialog is visible
const dialogVisible = ref(false)
// Whether in edit state
const isEdit = ref(false)

// Whether the assign-role dialog is visible
const allocDialogVisible = ref(false)
// Admin ID currently being assigned roles
const allocAdminId = ref<number>()
// Role IDs already assigned to the current admin
const allocRoleIds = ref<number[]>([])

// Get roles by admin ID
const getRoleListByAdmin = async (adminId: number) => {
  try {
    const res = await getRoleByAdminIdAPI(adminId)
    const allocRoleList = res.data
    allocRoleIds.value = []
    allocRoleList.forEach((item) => allocRoleIds.value.push(item.id!))
  } catch (error) {
    console.error('获取管理员角色列表失败:', error)
  }
}

// Reset search conditions
const handleResetSearch = () => {
  listQuery.value.pageNum = 1
  listQuery.value.keyword = ''
}

// Handle search
const handleSearchList = () => {
  listQuery.value.pageNum = 1
  getList()
}

// Page size change
const handleSizeChange = (val: number) => {
  listQuery.value.pageNum = 1
  listQuery.value.pageSize = val
  getList()
}

// Current page change
const handleCurrentChange = (val: number) => {
  listQuery.value.pageNum = val
  getList()
}

// Handle add admin
const handleAdd = () => {
  dialogVisible.value = true
  isEdit.value = false
  admin.value = {
    username: '',
    password: '',
    status: 1
  }
}

// Handle status change
const handleStatusChange = async (index: number, row: UmsAdmin) => {
  try {
    await ElMessageBox.confirm('是否要修改该状态?', '提示', {
      confirmButtonText: '确定',
      cancelButtonText: '取消',
      type: 'warning'
    })
    await adminUpdateStatusByIdAPI(row.id!, { status: row.status })
    ElMessage.success('修改成功!')
  } catch (error) {
    console.error('更新状态失败:', error)
    // If cancelled or failed, refetch the list to restore state
    getList()
  }
}

// Handle delete
const handleDelete = async (index: number, row: UmsAdmin) => {
  try {
    await ElMessageBox.confirm('是否要删除该用户?', '提示', {
      confirmButtonText: '确定',
      cancelButtonText: '取消',
      type: 'warning'
    })
    await adminDeleteByIdAPI(row.id!)
    ElMessage.success('删除成功!')
    getList()
  } catch (error) {
    console.error('删除失败:', error)
  }
}

// Handle update
const handleUpdate = (index: number, row: UmsAdmin) => {
  dialogVisible.value = true
  isEdit.value = true
  // Deep-copy the row object to avoid mutating the original data
  admin.value = Object.assign({}, row)
}

// Handle dialog confirmation
const handleDialogConfirm = async () => {
  try {
    await ElMessageBox.confirm('是否要确认?', '提示', {
      confirmButtonText: '确定',
      cancelButtonText: '取消',
      type: 'warning'
    })
    if (isEdit.value) {
      await adminUpdateByIdAPI(admin.value.id!, admin.value)
      ElMessage.success('修改成功！')
    } else {
      await adminRegisterAPI(admin.value)
      ElMessage.success('添加成功！')
    }
    dialogVisible.value = false
    getList()
  } catch (error) {
    console.error('操作失败:', error)
  }
}

// Handle assign-role dialog confirmation
const handleAllocDialogConfirm = async () => {
  try {
    await ElMessageBox.confirm('是否要确认?', '提示', {
      confirmButtonText: '确定',
      cancelButtonText: '取消',
      type: 'warning'
    })
    await adminRoleUpdateAPI({ adminId: allocAdminId.value!, roleIds: allocRoleIds.value.join(',') })
    ElMessage.success('分配成功！')
    allocDialogVisible.value = false
  } catch (error) {
    console.error('分配角色失败:', error)
  }
}

// Handle role selection
const handleSelectRole = (index: number, row: UmsAdmin) => {
  allocAdminId.value = row.id!
  allocDialogVisible.value = true
  getRoleListByAdmin(allocAdminId.value)
}

// Date format filter function
const formatDateTime = (time: string) => {
  if (!time) {
    return 'N/A'
  }
  return dayjs(time).format('YYYY-MM-DD HH:mm:ss')
}
</script>

<template>
  <div class="app-container">
    <el-card class="filter-container" shadow="never">
      <div>
        <el-icon class="el-icon-middle">
          <Search />
        </el-icon>
        <span>筛选搜索</span>
        <el-button style="float: right" @click="handleSearchList()" type="primary" size="default">
          查询搜索
        </el-button>
        <el-button style="float: right; margin-right: 15px" @click="handleResetSearch()" size="default">
          重置
        </el-button>
      </div>
      <div style="margin-top: 15px">
        <el-form :inline="true" :model="listQuery" size="default" label-width="140px">
          <el-form-item label="输入搜索：">
            <el-input v-model="listQuery.keyword" class="input-width" placeholder="帐号/姓名" clearable></el-input>
          </el-form-item>
        </el-form>
      </div>
    </el-card>
    <el-card class="operate-container" shadow="never">
      <el-icon class="el-icon-middle">
        <Tickets />
      </el-icon>
      <span>数据列表</span>
      <el-button class="btn-add" @click="handleAdd()">
        添加
      </el-button>
    </el-card>
    <div class="table-container">
      <el-table ref="adminTable" :data="list" style="width: 100%;" v-loading="listLoading" border>
        <el-table-column label="编号" width="100" align="center">
          <template #default="scope">{{ scope.row.id }}</template>
        </el-table-column>
        <el-table-column label="帐号" align="center">
          <template #default="scope">{{ scope.row.username }}</template>
        </el-table-column>
        <el-table-column label="姓名" align="center">
          <template #default="scope">{{ scope.row.nickName }}</template>
        </el-table-column>
        <el-table-column label="邮箱" align="center">
          <template #default="scope">{{ scope.row.email }}</template>
        </el-table-column>
        <el-table-column label="添加时间" width="160" align="center">
          <template #default="scope">{{ formatDateTime(scope.row.createTime) }}</template>
        </el-table-column>
        <el-table-column label="最后登录" width="160" align="center">
          <template #default="scope">{{ formatDateTime(scope.row.loginTime) }}</template>
        </el-table-column>
        <el-table-column label="是否启用" width="140" align="center">
          <template #default="scope">
            <el-switch @change="handleStatusChange(scope.$index, scope.row as UmsAdmin)" :active-value="1" :inactive-value="0"
              v-model="scope.row.status">
            </el-switch>
          </template>
        </el-table-column>
        <el-table-column label="操作" width="180" align="center">
          <template #default="scope">
            <el-button size="small" type="primary" link @click="handleSelectRole(scope.$index, scope.row as UmsAdmin)">分配角色
            </el-button>
            <el-button size="small" type="primary" link @click="handleUpdate(scope.$index, scope.row as UmsAdmin)">
              编辑
            </el-button>
            <el-button size="small" type="primary" link @click="handleDelete(scope.$index, scope.row as UmsAdmin)">删除
            </el-button>
          </template>
        </el-table-column>
      </el-table>
    </div>
    <div class="pagination-container">
      <el-pagination background @size-change="handleSizeChange" @current-change="handleCurrentChange"
        layout="total, sizes,prev, pager, next,jumper" v-model:current-page="listQuery.pageNum"
        :page-size="listQuery.pageSize" :page-sizes="[5, 10, 15]" :total="total">
      </el-pagination>
    </div>
    <el-dialog v-model="dialogVisible" :title="isEdit ? '编辑用户' : '添加用户'" width="40%">
      <el-form :model="admin" label-width="150px" size="default">
        <el-form-item label="帐号：">
          <el-input v-model="admin.username" style="width: 250px"></el-input>
        </el-form-item>
        <el-form-item label="姓名：">
          <el-input v-model="admin.nickName" style="width: 250px"></el-input>
        </el-form-item>
        <el-form-item label="邮箱：">
          <el-input v-model="admin.email" style="width: 250px"></el-input>
        </el-form-item>
        <el-form-item label="密码：" v-if="!isEdit">
          <el-input v-model="admin.password" type="password" style="width: 250px"></el-input>
        </el-form-item>
        <el-form-item label="备注：">
          <el-input v-model="admin.note" type="textarea" :rows="5" style="width: 250px"></el-input>
        </el-form-item>
        <el-form-item label="是否启用：">
          <el-radio-group v-model="admin.status">
            <el-radio :label="1">是</el-radio>
            <el-radio :label="0">否</el-radio>
          </el-radio-group>
        </el-form-item>
      </el-form>
      <template #footer>
        <span class="dialog-footer">
          <el-button @click="dialogVisible = false" size="default">取 消</el-button>
          <el-button type="primary" @click="handleDialogConfirm()" size="default">确 定</el-button>
        </span>
      </template>
    </el-dialog>
    <el-dialog title="分配角色" v-model="allocDialogVisible" width="30%">
      <el-select v-model="allocRoleIds" multiple placeholder="请选择" size="default" style="width: 80%">
        <el-option v-for="item in allRoleList" :key="item.id" :label="item.name" :value="item.id!">
        </el-option>
      </el-select>
      <template #footer>
        <span class="dialog-footer">
          <el-button @click="allocDialogVisible = false" size="default">取 消</el-button>
          <el-button type="primary" @click="handleAllocDialogConfirm()" size="default">确 定</el-button>
        </span>
      </template>
    </el-dialog>
  </div>
</template>

<style scoped></style>

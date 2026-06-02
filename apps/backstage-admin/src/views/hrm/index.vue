<script setup lang="ts">
import { ref, onMounted, computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Search, Plus, Edit, View, Delete, Refresh, User, OfficeBuilding } from '@element-plus/icons-vue'
import {
  getAllDepartments,
  getDepartmentTree,
  createDepartment,
  updateDepartment,
  deleteDepartment,
  getAllEmployees,
  getEmployeesByDepartment,
  createEmployee,
  updateEmployee,
  deleteEmployee,
  type Department,
  type DepartmentCreateCommand,
  type DepartmentUpdateCommand,
  type Employee,
  type EmployeeCreateCommand,
  type EmployeeUpdateCommand
} from '@/apis/hrm'

const { t } = useI18n()

// Tabs
const activeTab = ref('department')

// Department
const deptList = ref<Department[]>([])
const deptTree = ref<Department[]>([])
const deptLoading = ref(false)
const deptDialogVisible = ref(false)
const deptDialogLoading = ref(false)
const isDeptEdit = ref(false)
const deptFormData = ref<Department>({
  id: undefined,
  departmentCode: '',
  departmentName: '',
  parentId: null,
  parentName: '',
  level: 1,
  leaderId: null,
  leaderName: '',
  phone: '',
  email: '',
  status: 1,
  sort: 0,
  remark: ''
})

// Employee
const empList = ref<Employee[]>([])
const empLoading = ref(false)
const empDialogVisible = ref(false)
const empDialogLoading = ref(false)
const isEmpEdit = ref(false)
const selectedDeptId = ref<number | null>(null)
const empFormData = ref<Employee>({
  id: undefined,
  employeeNo: '',
  name: '',
  gender: 1,
  birthday: '',
  idCard: '',
  mobile: '',
  email: '',
  departmentId: 0,
  departmentName: '',
  positionId: 0,
  positionName: '',
  employeeStatus: 1,
  employeeType: 1,
  entryDate: '',
  contractStartDate: '',
  contractEndDate: '',
  baseSalary: 0,
  bankName: '',
  bankAccount: '',
  emergencyContact: '',
  emergencyPhone: '',
  address: '',
  remark: ''
})

// Status options
const statusOptions = [
  { label: 'Active', value: 1 },
  { label: 'Inactive', value: 0 }
]

const genderOptions = [
  { label: 'Male', value: 1 },
  { label: 'Female', value: 2 }
]

const employeeStatusOptions = [
  { label: 'Probation', value: 1 },
  { label: 'Formal', value: 2 },
  { label: 'Resigned', value: 3 }
]

const employeeTypeOptions = [
  { label: 'Full-time', value: 1 },
  { label: 'Part-time', value: 2 },
  { label: 'Contract', value: 3 }
]

// Department methods
const loadDepartments = async () => {
  deptLoading.value = true
  try {
    const res = await getAllDepartments()
    deptList.value = res.data || []
  } catch (error) {
    ElMessage.error('Failed to load departments')
  } finally {
    deptLoading.value = false
  }
}

const loadDepartmentTree = async () => {
  deptLoading.value = true
  try {
    const res = await getDepartmentTree()
    deptTree.value = res.data || []
  } catch (error) {
    ElMessage.error('Failed to load department tree')
  } finally {
    deptLoading.value = false
  }
}

const handleAddDepartment = () => {
  isDeptEdit.value = false
  deptFormData.value = {
    id: undefined,
    departmentCode: '',
    departmentName: '',
    parentId: null,
    parentName: '',
    level: 1,
    leaderId: null,
    leaderName: '',
    phone: '',
    email: '',
    status: 1,
    sort: 0,
    remark: ''
  }
  deptDialogVisible.value = true
}

const handleEditDepartment = (row: Department) => {
  isDeptEdit.value = true
  deptFormData.value = { ...row }
  deptDialogVisible.value = true
}

const handleDeleteDepartment = async (row: Department) => {
  if (!row.id) return
  try {
    await ElMessageBox.confirm('Delete this department?', 'Confirm', { type: 'warning' })
    await deleteDepartment(row.id)
    ElMessage.success('Department deleted')
    loadDepartmentTree()
  } catch (error) {
    if (error !== 'cancel') {
      ElMessage.error('Failed to delete department')
    }
  }
}

const handleSubmitDepartment = async () => {
  deptDialogLoading.value = true
  try {
    if (isDeptEdit.value && deptFormData.value.id) {
      const data: DepartmentUpdateCommand = {
        id: deptFormData.value.id,
        departmentName: deptFormData.value.departmentName,
        parentId: deptFormData.value.parentId,
        level: deptFormData.value.level,
        leaderId: deptFormData.value.leaderId,
        leaderName: deptFormData.value.leaderName,
        phone: deptFormData.value.phone,
        email: deptFormData.value.email,
        status: deptFormData.value.status,
        sort: deptFormData.value.sort,
        remark: deptFormData.value.remark
      }
      await updateDepartment(data)
      ElMessage.success('Department updated')
    } else {
      const data: DepartmentCreateCommand = {
        departmentCode: deptFormData.value.departmentCode,
        departmentName: deptFormData.value.departmentName,
        parentId: deptFormData.value.parentId,
        level: deptFormData.value.level,
        leaderId: deptFormData.value.leaderId,
        leaderName: deptFormData.value.leaderName,
        phone: deptFormData.value.phone,
        email: deptFormData.value.email,
        status: deptFormData.value.status,
        sort: deptFormData.value.sort,
        remark: deptFormData.value.remark
      }
      await createDepartment(data)
      ElMessage.success('Department created')
    }
    deptDialogVisible.value = false
    loadDepartmentTree()
  } catch (error) {
    ElMessage.error(isDeptEdit.value ? 'Failed to update department' : 'Failed to create department')
  } finally {
    deptDialogLoading.value = false
  }
}

// Employee methods
const loadEmployees = async () => {
  empLoading.value = true
  try {
    let res
    if (selectedDeptId.value) {
      res = await getEmployeesByDepartment(selectedDeptId.value)
    } else {
      res = await getAllEmployees()
    }
    empList.value = res.data || []
  } catch (error) {
    ElMessage.error('Failed to load employees')
  } finally {
    empLoading.value = false
  }
}

const handleAddEmployee = () => {
  isEmpEdit.value = false
  empFormData.value = {
    id: undefined,
    employeeNo: '',
    name: '',
    gender: 1,
    birthday: '',
    idCard: '',
    mobile: '',
    email: '',
    departmentId: selectedDeptId.value || 0,
    departmentName: '',
    positionId: 0,
    positionName: '',
    employeeStatus: 1,
    employeeType: 1,
    entryDate: '',
    contractStartDate: '',
    contractEndDate: '',
    baseSalary: 0,
    bankName: '',
    bankAccount: '',
    emergencyContact: '',
    emergencyPhone: '',
    address: '',
    remark: ''
  }
  empDialogVisible.value = true
}

const handleEditEmployee = (row: Employee) => {
  isEmpEdit.value = true
  empFormData.value = { ...row }
  empDialogVisible.value = true
}

const handleDeleteEmployee = async (row: Employee) => {
  if (!row.id) return
  try {
    await ElMessageBox.confirm('Delete this employee?', 'Confirm', { type: 'warning' })
    await deleteEmployee(row.id)
    ElMessage.success('Employee deleted')
    loadEmployees()
  } catch (error) {
    if (error !== 'cancel') {
      ElMessage.error('Failed to delete employee')
    }
  }
}

const handleSubmitEmployee = async () => {
  empDialogLoading.value = true
  try {
    if (isEmpEdit.value && empFormData.value.id) {
      const data: EmployeeUpdateCommand = {
        id: empFormData.value.id,
        name: empFormData.value.name,
        gender: empFormData.value.gender,
        birthday: empFormData.value.birthday,
        idCard: empFormData.value.idCard,
        mobile: empFormData.value.mobile,
        email: empFormData.value.email,
        departmentId: empFormData.value.departmentId,
        departmentName: empFormData.value.departmentName,
        positionId: empFormData.value.positionId,
        positionName: empFormData.value.positionName,
        employeeStatus: empFormData.value.employeeStatus,
        employeeType: empFormData.value.employeeType,
        entryDate: empFormData.value.entryDate,
        contractStartDate: empFormData.value.contractStartDate,
        contractEndDate: empFormData.value.contractEndDate,
        baseSalary: empFormData.value.baseSalary,
        bankName: empFormData.value.bankName,
        bankAccount: empFormData.value.bankAccount,
        emergencyContact: empFormData.value.emergencyContact,
        emergencyPhone: empFormData.value.emergencyPhone,
        address: empFormData.value.address,
        remark: empFormData.value.remark
      }
      await updateEmployee(data)
      ElMessage.success('Employee updated')
    } else {
      const data: EmployeeCreateCommand = {
        employeeNo: empFormData.value.employeeNo,
        name: empFormData.value.name,
        gender: empFormData.value.gender,
        birthday: empFormData.value.birthday,
        idCard: empFormData.value.idCard,
        mobile: empFormData.value.mobile,
        email: empFormData.value.email,
        departmentId: empFormData.value.departmentId,
        departmentName: empFormData.value.departmentName,
        positionId: empFormData.value.positionId,
        positionName: empFormData.value.positionName,
        employeeStatus: empFormData.value.employeeStatus,
        employeeType: empFormData.value.employeeType,
        entryDate: empFormData.value.entryDate,
        contractStartDate: empFormData.value.contractStartDate,
        contractEndDate: empFormData.value.contractEndDate,
        baseSalary: empFormData.value.baseSalary,
        bankName: empFormData.value.bankName,
        bankAccount: empFormData.value.bankAccount,
        emergencyContact: empFormData.value.emergencyContact,
        emergencyPhone: empFormData.value.emergencyPhone,
        address: empFormData.value.address,
        remark: empFormData.value.remark
      }
      await createEmployee(data)
      ElMessage.success('Employee created')
    }
    empDialogVisible.value = false
    loadEmployees()
  } catch (error) {
    ElMessage.error(isEmpEdit.value ? 'Failed to update employee' : 'Failed to create employee')
  } finally {
    empDialogLoading.value = false
  }
}

const handleDeptChange = (deptId: number | null) => {
  selectedDeptId.value = deptId
  loadEmployees()
}

// Format helpers
const getStatusType = (status: number): 'success' | 'warning' | 'info' | 'danger' | undefined => {
  const map: Record<number, 'success' | 'warning' | 'info' | 'danger'> = {
    1: 'success',
    0: 'info'
  }
  return map[status]
}

const getGenderLabel = (gender: number): string => {
  const map: Record<number, string> = {
    1: 'Male',
    2: 'Female'
  }
  return map[gender] || String(gender)
}

const getEmployeeStatusLabel = (status: number): string => {
  const map: Record<number, string> = {
    1: 'Probation',
    2: 'Formal',
    3: 'Resigned'
  }
  return map[status] || String(status)
}

const getEmployeeTypeLabel = (type: number): string => {
  const map: Record<number, string> = {
    1: 'Full-time',
    2: 'Part-time',
    3: 'Contract'
  }
  return map[type] || String(type)
}

onMounted(() => {
  loadDepartmentTree()
  loadEmployees()
})
</script>

<template>
  <div class="hrm-container">
    <el-tabs v-model="activeTab" type="border-card">
      <el-tab-pane :label="t('hrm.department')" name="department">
        <el-card class="search-card">
          <div class="toolbar">
            <el-button type="primary" :icon="Plus" @click="handleAddDepartment">{{ t('common.add') }}</el-button>
            <el-button :icon="Refresh" @click="loadDepartmentTree">{{ t('common.refresh') }}</el-button>
          </div>
        </el-card>

        <el-card class="table-card">
          <el-table v-loading="deptLoading" :data="deptTree" border stripe row-key="id" default-expand-all>
            <el-table-column prop="departmentCode" :label="t('hrm.departmentCode')" width="150" />
            <el-table-column prop="departmentName" :label="t('hrm.departmentName')" min-width="180" />
            <el-table-column prop="parentName" :label="t('hrm.parentDepartment')" width="150" />
            <el-table-column prop="level" :label="t('hrm.level')" width="80" />
            <el-table-column prop="leaderName" :label="t('hrm.leader')" width="120" />
            <el-table-column prop="phone" :label="t('hrm.phone')" width="130" />
            <el-table-column prop="email" :label="t('hrm.email')" width="180" />
            <el-table-column prop="status" :label="t('hrm.status')" width="100">
              <template #default="{ row }">
                <el-tag :type="getStatusType(row.status)">{{ row.status === 1 ? 'Active' : 'Inactive' }}</el-tag>
              </template>
            </el-table-column>
            <el-table-column prop="sort" :label="t('hrm.sortOrder')" width="80" />
            <el-table-column :label="t('common.actions')" width="150" fixed="right">
              <template #default="{ row }">
                <el-button link type="primary" size="small" @click="handleEditDepartment(row)">{{ t('common.edit') }}</el-button>
                <el-button link type="danger" size="small" @click="handleDeleteDepartment(row)">{{ t('common.delete') }}</el-button>
              </template>
            </el-table-column>
          </el-table>
        </el-card>

        <el-dialog v-model="deptDialogVisible" :title="isDeptEdit ? t('hrm.editDepartment') : t('hrm.addDepartment')" width="600px">
          <el-form v-loading="deptDialogLoading" :model="deptFormData" label-width="140px">
            <el-form-item :label="t('hrm.departmentCode')">
              <el-input v-model="deptFormData.departmentCode" :disabled="isDeptEdit" />
            </el-form-item>
            <el-form-item :label="t('hrm.departmentName')">
              <el-input v-model="deptFormData.departmentName" />
            </el-form-item>
            <el-form-item :label="t('hrm.parentDepartment')">
              <el-select v-model="deptFormData.parentId" clearable placeholder="Root Department">
                <el-option v-for="dept in deptList" :key="dept.id" :label="dept.departmentName" :value="dept.id!" />
              </el-select>
            </el-form-item>
            <el-form-item :label="t('hrm.level')">
              <el-input-number v-model="deptFormData.level" :min="1" :max="10" />
            </el-form-item>
            <el-form-item :label="t('hrm.leader')">
              <el-input v-model="deptFormData.leaderName" />
            </el-form-item>
            <el-form-item :label="t('hrm.phone')">
              <el-input v-model="deptFormData.phone" />
            </el-form-item>
            <el-form-item :label="t('hrm.email')">
              <el-input v-model="deptFormData.email" />
            </el-form-item>
            <el-form-item :label="t('hrm.status')">
              <el-select v-model="deptFormData.status">
                <el-option v-for="item in statusOptions" :key="item.value" :label="item.label" :value="item.value" />
              </el-select>
            </el-form-item>
            <el-form-item :label="t('hrm.sortOrder')">
              <el-input-number v-model="deptFormData.sort" :min="0" />
            </el-form-item>
            <el-form-item :label="t('hrm.remark')">
              <el-input v-model="deptFormData.remark" type="textarea" :rows="3" />
            </el-form-item>
          </el-form>
          <template #footer>
            <el-button @click="deptDialogVisible = false">{{ t('common.cancel') }}</el-button>
            <el-button type="primary" @click="handleSubmitDepartment">{{ t('common.submit') }}</el-button>
          </template>
        </el-dialog>
      </el-tab-pane>

      <el-tab-pane :label="t('hrm.employee')" name="employee">
        <el-card class="search-card">
          <el-form :inline="true" :model="{}" class="search-form">
            <el-form-item :label="t('hrm.department')">
              <el-select :model-value="selectedDeptId" @change="handleDeptChange" clearable placeholder="All Departments">
                <el-option v-for="dept in deptList" :key="dept.id" :label="dept.departmentName" :value="dept.id!" />
              </el-select>
            </el-form-item>
            <el-form-item>
              <el-button type="primary" :icon="Refresh" @click="loadEmployees">{{ t('common.refresh') }}</el-button>
            </el-form-item>
          </el-form>
        </el-card>

        <el-card class="table-card">
          <div class="toolbar">
            <el-button type="primary" :icon="Plus" @click="handleAddEmployee">{{ t('common.add') }}</el-button>
          </div>

          <el-table v-loading="empLoading" :data="empList" border stripe>
            <el-table-column prop="employeeNo" :label="t('hrm.employeeNo')" width="120" />
            <el-table-column prop="name" :label="t('hrm.name')" width="100" />
            <el-table-column prop="gender" :label="t('hrm.gender')" width="80">
              <template #default="{ row }">
                {{ getGenderLabel(row.gender) }}
              </template>
            </el-table-column>
            <el-table-column prop="mobile" :label="t('hrm.mobile')" width="130" />
            <el-table-column prop="email" :label="t('hrm.email')" width="180" />
            <el-table-column prop="departmentName" :label="t('hrm.department')" width="120" />
            <el-table-column prop="positionName" :label="t('hrm.position')" width="120" />
            <el-table-column prop="employeeStatus" :label="t('hrm.status')" width="100">
              <template #default="{ row }">
                <el-tag :type="row.employeeStatus === 2 ? 'success' : (row.employeeStatus === 1 ? 'warning' : 'info')">
                  {{ getEmployeeStatusLabel(row.employeeStatus) }}
                </el-tag>
              </template>
            </el-table-column>
            <el-table-column prop="employeeType" :label="t('hrm.employeeType')" width="100">
              <template #default="{ row }">
                {{ getEmployeeTypeLabel(row.employeeType) }}
              </template>
            </el-table-column>
            <el-table-column prop="entryDate" :label="t('hrm.entryDate')" width="120" />
            <el-table-column :label="t('common.actions')" width="150" fixed="right">
              <template #default="{ row }">
                <el-button link type="primary" size="small" @click="handleEditEmployee(row)">{{ t('common.edit') }}</el-button>
                <el-button link type="danger" size="small" @click="handleDeleteEmployee(row)">{{ t('common.delete') }}</el-button>
              </template>
            </el-table-column>
          </el-table>
        </el-card>

        <el-dialog v-model="empDialogVisible" :title="isEmpEdit ? t('hrm.editEmployee') : t('hrm.addEmployee')" width="700px">
          <el-form v-loading="empDialogLoading" :model="empFormData" label-width="120px">
            <el-row :gutter="20">
              <el-col :span="12">
                <el-form-item :label="t('hrm.employeeNo')">
                  <el-input v-model="empFormData.employeeNo" :disabled="isEmpEdit" />
                </el-form-item>
              </el-col>
              <el-col :span="12">
                <el-form-item :label="t('hrm.name')">
                  <el-input v-model="empFormData.name" />
                </el-form-item>
              </el-col>
            </el-row>
            <el-row :gutter="20">
              <el-col :span="12">
                <el-form-item :label="t('hrm.gender')">
                  <el-select v-model="empFormData.gender">
                    <el-option v-for="item in genderOptions" :key="item.value" :label="item.label" :value="item.value" />
                  </el-select>
                </el-form-item>
              </el-col>
              <el-col :span="12">
                <el-form-item :label="t('hrm.birthday')">
                  <el-date-picker v-model="empFormData.birthday" type="date" value-format="YYYY-MM-DD" style="width: 100%" />
                </el-form-item>
              </el-col>
            </el-row>
            <el-row :gutter="20">
              <el-col :span="12">
                <el-form-item :label="t('hrm.mobile')">
                  <el-input v-model="empFormData.mobile" />
                </el-form-item>
              </el-col>
              <el-col :span="12">
                <el-form-item :label="t('hrm.email')">
                  <el-input v-model="empFormData.email" />
                </el-form-item>
              </el-col>
            </el-row>
            <el-row :gutter="20">
              <el-col :span="12">
                <el-form-item :label="t('hrm.idCard')">
                  <el-input v-model="empFormData.idCard" />
                </el-form-item>
              </el-col>
              <el-col :span="12">
                <el-form-item :label="t('hrm.department')">
                  <el-select v-model="empFormData.departmentId" @change="(val: number) => {
                    const dept = deptList.find(d => d.id === val)
                    if (dept) empFormData.departmentName = dept.departmentName
                  }">
                    <el-option v-for="dept in deptList" :key="dept.id" :label="dept.departmentName" :value="dept.id!" />
                  </el-select>
                </el-form-item>
              </el-col>
            </el-row>
            <el-row :gutter="20">
              <el-col :span="12">
                <el-form-item :label="t('hrm.position')">
                  <el-input v-model="empFormData.positionName" />
                </el-form-item>
              </el-col>
              <el-col :span="12">
                <el-form-item :label="t('hrm.status')">
                  <el-select v-model="empFormData.employeeStatus">
                    <el-option v-for="item in employeeStatusOptions" :key="item.value" :label="item.label" :value="item.value" />
                  </el-select>
                </el-form-item>
              </el-col>
            </el-row>
            <el-row :gutter="20">
              <el-col :span="12">
                <el-form-item :label="t('hrm.employeeType')">
                  <el-select v-model="empFormData.employeeType">
                    <el-option v-for="item in employeeTypeOptions" :key="item.value" :label="item.label" :value="item.value" />
                  </el-select>
                </el-form-item>
              </el-col>
              <el-col :span="12">
                <el-form-item :label="t('hrm.entryDate')">
                  <el-date-picker v-model="empFormData.entryDate" type="date" value-format="YYYY-MM-DD" style="width: 100%" />
                </el-form-item>
              </el-col>
            </el-row>
            <el-row :gutter="20">
              <el-col :span="12">
                <el-form-item :label="t('hrm.contractStart')">
                  <el-date-picker v-model="empFormData.contractStartDate" type="date" value-format="YYYY-MM-DD" style="width: 100%" />
                </el-form-item>
              </el-col>
              <el-col :span="12">
                <el-form-item :label="t('hrm.contractEnd')">
                  <el-date-picker v-model="empFormData.contractEndDate" type="date" value-format="YYYY-MM-DD" style="width: 100%" />
                </el-form-item>
              </el-col>
            </el-row>
            <el-row :gutter="20">
              <el-col :span="12">
                <el-form-item :label="t('hrm.baseSalary')">
                  <el-input-number v-model="empFormData.baseSalary" :min="0" :precision="2" style="width: 100%" />
                </el-form-item>
              </el-col>
              <el-col :span="12">
                <el-form-item :label="t('hrm.bankName')">
                  <el-input v-model="empFormData.bankName" />
                </el-form-item>
              </el-col>
            </el-row>
            <el-row :gutter="20">
              <el-col :span="12">
                <el-form-item :label="t('hrm.bankAccount')">
                  <el-input v-model="empFormData.bankAccount" />
                </el-form-item>
              </el-col>
              <el-col :span="12">
                <el-form-item :label="t('hrm.address')">
                  <el-input v-model="empFormData.address" />
                </el-form-item>
              </el-col>
            </el-row>
            <el-row :gutter="20">
              <el-col :span="12">
                <el-form-item :label="t('hrm.emergencyContact')">
                  <el-input v-model="empFormData.emergencyContact" />
                </el-form-item>
              </el-col>
              <el-col :span="12">
                <el-form-item :label="t('hrm.emergencyPhone')">
                  <el-input v-model="empFormData.emergencyPhone" />
                </el-form-item>
              </el-col>
            </el-row>
            <el-form-item :label="t('hrm.remark')">
              <el-input v-model="empFormData.remark" type="textarea" :rows="2" />
            </el-form-item>
          </el-form>
          <template #footer>
            <el-button @click="empDialogVisible = false">{{ t('common.cancel') }}</el-button>
            <el-button type="primary" @click="handleSubmitEmployee">{{ t('common.submit') }}</el-button>
          </template>
        </el-dialog>
      </el-tab-pane>
    </el-tabs>
  </div>
</template>

<style scoped>
.hrm-container {
  padding: 20px;
}

.search-card {
  margin-bottom: 20px;
}

.table-card {
  margin-bottom: 20px;
}

.toolbar {
  display: flex;
  gap: 10px;
  margin-bottom: 15px;
}

.search-form {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
}
</style>
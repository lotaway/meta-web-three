import type { UmsMenu } from './menu'

/** Login request params */
export type LoginParam = {
  /** Username */
  username: string
  /** Password */
  password: string
}

/** Login response result */
export type LoginResult = {
  /** Token prefix */
  tokenHead: string
  /** Login token */
  token: string
}

/** User info result wrapper */
export type UserInfoResult = {
  /** Username */
  username: string
  /** Avatar */
  icon: string
  /** Menu */
  menus: UmsMenu[]
  /** Role */
  roles: []
}

/** User info (stored in store) */
export type UserInfo = Pick<UserInfoResult, 'username' | 'menus' | 'roles'> & {
  /** Login token */
  token: string
  /** Avatar */
  avatar
}

/** Admin info */
export type UmsAdmin = {
  /** ID */
  id?: number
  /** Username */
  username: string
  /** Password */
  password: string
  /** Avatar */
  icon?: string
  /** Email */
  email?: string
  /** Nickname */
  nickName?: string
  /** Remark */
  note?: string
  /** Creation time */
  createTime?: string
  /** Last login time */
  loginTime?: string
  /** Account status: 0 -> disabled; 1 -> enabled */
  status: number
}

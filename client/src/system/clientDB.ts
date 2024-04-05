export namespace ClientStore {

    function test() {
        const upgradeDBScript = new UpgradeDBScript(1)
        const clientDBTable = new ClientDBTable("nfts")
        clientDBTable
            .addField(new ClientDBField({
                name: "blockNumber",
                unique: true,
            }))
            .addField(new ClientDBField({
                name: "txid",
                unique: true,
            }))
            .addFieldsAsSimple(["startTime", "endTime", "image", "name", "price", "introduct", "description", "holderCount"])
        upgradeDBScript.addCommand(clientDBTable)
    }

    export class ClientDB {

        protected upgradeScripts: UpgradeDBScript[] = []
        protected db: IDBDatabase | null = null

        constructor(readonly dbFactory: IDBFactory) {

        }

        addUprade(script: UpgradeDBScript) {
            this.upgradeScripts.push(script)
        }

        connect(dbName: string, version: number = 1) {
            const self = this
            const request = this.dbFactory.open(dbName, version)
            return new Promise(function (resolve, reject) {
                request.onupgradeneeded = function (event) {
                    const db = request.result
                    self.upgradeScripts.sort((first, second) => first.version - second.version).forEach(script => {
                        if (event.oldVersion < script.version) {
                            script.use(db)
                        }
                    })
                }
                request.onsuccess = function (event) {
                    self.db = request.result
                    resolve(self.db)
                }
                request.onerror = reject
            })
        }

        close() {
            return this.db?.close?.()
        }

        getTable(storeName: string) {
            return new ClientDBTable(storeName)
        }

    }

    export class UpgradeDBScript {

        protected commands: DBCommand[] = []

        constructor(readonly version: number) {

        }

        addCommand(command: DBCommand) {
            this.commands.push(command)
        }

        use(db: IDBDatabase) {
            for (const command of this.commands) {
                command.use(db)
            }
        }

    }

    export interface DBCommand {
        use(db: IDBDatabase): void
    }

    export interface IPageWrapper<DAO extends object> {
        getParams(): Partial<DAO>
        getSelector?: (params: Partial<DAO>) => IDBRequest<IDBCursorWithValue | null>
        getPage(): number
        getPageSize(): number
        getRange(): IDBKeyRange
        handler(data: DAO, cursor: IDBCursor): boolean
    }

    export abstract class BasePageWrapper<VO extends object> implements IPageWrapper<VO> {
        getPage(): number {
            return 1
        }
        getPageSize(): number {
            return 20
        }
        getRange(): IDBKeyRange {
            return IDBKeyRange.only('id')
        }
        abstract getParams(): Partial<VO>
        abstract handler(data: VO, cursor: IDBCursor): boolean
    }

    export class ClientDBTable<DO extends object> implements DBCommand {

        protected fields: ClientDBField[] = []

        constructor(readonly name: string, readonly key: string = "id") {

        }

        addField(field: ClientDBField) {
            if (this.key === field.name) {
                throw new Error(`New field can't have the same as key name: ${this.key}`)
            }
            this.fields.push(field)
            return this
        }

        addFieldsAsSimple(names: string[]) {
            names.forEach(name => {
                this.addField(new ClientDBField({
                    name,
                }))
            })
            return this
        }

        use(db: IDBDatabase): IDBObjectStore {
            const objectStore = db.createObjectStore(this.name, {
                keyPath: this.key,
            })
            for (const field of this.fields) {
                field.use(objectStore)
            }
            return objectStore
        }

        getData(db: IDBDatabase) {
            const transaction = db.transaction(this.name, "readonly")
            const store = transaction.objectStore(this.name)
            const request = store.getAll()
            return new Promise((resolve, reject) => {
                request.onsuccess = resolve
                request.onerror = reject
            })
        }

        queryByPage(store: IDBObjectStore, wrapper: IPageWrapper<Partial<DO>>) {
            const page = wrapper.getPage()
            const pageSize = wrapper.getPageSize()
            return new Promise((resolve, reject) => {
                let needSkip = true
                const request = wrapper.getSelector?.(wrapper.getParams()) ?? store.openCursor(wrapper.getRange())
                request.onsuccess = function (event) {
                    const cursor = request.result
                    if (!cursor || !cursor.value) {
                        return resolve(true)
                    }
                    if (needSkip) {
                        cursor.advance((page - 1) * pageSize)
                        needSkip = false
                    }
                    if (wrapper.handler(cursor.value, cursor)) {
                        return cursor.continue()
                    }
                }
                request.onerror = reject
            })
        }

        async getDataByPage(store: IDBObjectStore, params: Partial<DO>) {
            const tableRows: DO[] = []
            const wrapper = new class extends BasePageWrapper<DO> {
                getParams(): Partial<DO> {
                    return params
                }
                handler(data: DO): boolean {
                    tableRows.push(data)
                    return tableRows.length < this.getPageSize()
                }

            }
            await this.queryByPage(store, wrapper)
            return tableRows
        }

        addData(store: IDBObjectStore, data: DO) {
            return new Promise((resolve, reject) => {
                const request = store.add(data)
                request.onsuccess = resolve
                request.onerror = reject
            })
        }

        updateData(store: IDBObjectStore, data: DO) {
            return new Promise((resolve, reject) => {
                const request = store.put(data)
                request.onsuccess = resolve
                request.onerror = reject
            })
        }

        delData(store: IDBObjectStore, value: string) {
            return new Promise((resolve, reject) => {
                const request = store.delete(IDBKeyRange.only(value))
                request.onsuccess = resolve
                request.onerror = reject
            })
        }

        delDatas(store: IDBObjectStore, key: keyof DO, value: string) {
            return new Promise(async (resolve, reject) => {
                const wrapper = new class extends BasePageWrapper<DO> {
                    getSelector() {
                        return store.index(key as string).openCursor(IDBKeyRange.only(value))
                    }
                    getParams(): Partial<DO> {
                        return {
                            [key]: value,
                        } as Partial<DO>
                    }
                    handler(data: DO, cursor: IDBCursor): boolean {
                        const request = cursor.delete()
                        request.onsuccess = function () {
                            if (request.result !== undefined)
                                reject("delete fail")
                        }
                        request.onerror = reject
                        return true
                    }
                }
                await this.queryByPage(store, wrapper)
            })
        }
    }

    export class ClientDBField<Type = string> {

        constructor(readonly data: {
            name: string
            key?: string
            unique?: boolean
        }) {
            this.data.unique = this.data.unique ?? false
        }

        get name() {
            return this.data.name
        }

        use(objectStore: IDBObjectStore) {
            objectStore.createIndex(this.data.name, this.data.key ?? this.data.name, {
                unique: this.data.unique,
            })
        }
    }
}
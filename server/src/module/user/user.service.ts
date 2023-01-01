import {Injectable} from "@nestjs/common";

interface User extends Object {

}

@Injectable()
export class UserService {
    private readonly users: User[];

    getUsers() {
        return this.users;
    }

    addFollower() {

    }

    getFollowers() {
        return [];
    }

    getUserBlogCount() {
        return [];
    }
}
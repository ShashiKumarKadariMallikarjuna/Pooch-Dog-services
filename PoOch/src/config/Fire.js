import firebase from "firebase";

var config = {
  apiKey: "AIzaSyArtX2gJiOhztjfqNQx2nqbXt-F-s7jTGs",
  authDomain: "cecs-491-1934c.firebaseapp.com",
  databaseURL: "https://cecs-491-1934c.firebaseio.com",
  projectId: "cecs-491-1934c",
  storageBucket: "cecs-491-1934c.appspot.com",
  messagingSenderId: "85110920735",
  appId: "1:85110920735:web:dfd9f1e95c6042c9e0672a"
};

const fire = firebase.initializeApp(config);
export const googleProvider = new firebase.auth.GoogleAuthProvider();
export default fire;

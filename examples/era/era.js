const a = new Array(1000);
let anum = 0;
const aflag = new Array(1001);

for (let i = 2; i <= 1000; i++) {
  if (aflag[i] == null) {
    a[anum++] = i;
    for (let j = i * i; j <= 1000; j += i) {
      aflag[j] = 1;
    }
  }
}

console.log(a)
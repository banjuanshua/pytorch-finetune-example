import sys
res = []
while True:
    n_num = sys.stdin.readline().strip()
    if not n_num:
        break
    n_num = list(map(int,n_num.split()))
    n_list = list(map(int,sys.stdin.readline().strip().split()))

    allnum = sum(n_list)
    if allnum%n_num[1] == 0:
        need = allnum//n_num[1]
    else:
        need = (allnum//n_num[1])+1
    if n_num[2]*60 > need:
        res.append(need)
    else:
        if allnum-n_num[2]<=480:
            res.append(allnum-n_num[2])
        else:
            res.append(0)
    for i in res:
        print(i)




import java.util.*;

public class Main {
    public static void main(String[] args) {
        Scanner sc=new Scanner(System.in);
        while (sc.hasNextInt()){
            int sum=0;
            int n = sc.nextInt();
            int p= sc.nextInt();
            int c=sc.nextInt();
            int[] dp=new int[n];
            for (int i=0;i<dp.length;i++){
                dp[i]=Integer.MAX_VALUE;
            }
            dp[0]=0;
            List<Way> list=new ArrayList<>();
            for (int i=0;i<p;i++){
                list.add(new Way(sc.nextInt(),sc.nextInt(),sc.nextInt()));
            }
            Collections.sort(list, new Comparator<Way>() {
                @Override
                public int compare(Way o1, Way o2) {
                    if (o1.start!=o2.start){
                        return o1.start-o2.start;
                    }else {
                        return o1.end-o2.end;
                    }
                }
            });
            for (int i=0;i<p;i++){
                Way way = list.get(i);
                dp[way.end]=Math.min(dp[way.end],dp[way.start]+way.time);
            }
            int[] tmp=new int[c];
            for (int i=0;i<c;i++){
                tmp[i]=sc.nextInt();
            }
            for (int i=0;i<c;i++){
                sum=sum+dp[tmp[i]];
            }
            System.out.println(sum);
        }
    }
    static class Way{
        int start;
        int end;
        int time;

        public Way(int start, int end, int time) {
            this.start = start;
            this.end = end;
            this.time = time;
        }
    }
}

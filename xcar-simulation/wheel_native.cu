struct Wheel {
    float vx, denom, radius, zy, sy, bx, cx, ex, peak, fz, blend, inertia, old, resist;
};
struct Motor {
    float gear, eff, demand, alpha, coast, state, battery, cap, torque_ref, exponent;
};
__device__ float lookup(float x, const float* grid, const float* values, int n) {
    int lo=1, hi=n-1;
    while(lo<hi) {
        int mid=(lo+hi)/2;
        if(grid[mid]<x) lo=mid+1; else hi=mid;
    }
    return values[lo-1]+(values[lo]-values[lo-1])*(x-grid[lo-1])/(grid[lo]-grid[lo-1]);
}
__device__ float shape(float x, Wheel w) {
    float t=w.bx*x;
    return sinf(w.cx*atanf(t-w.ex*(t-atanf(t))));
}
__device__ float residual(float v, Wheel w, Motor m, int lane, unsigned mask, const float* rpm,
    const float* trq, int nm, const float* brpm, const float* btrq, int nb,
    float kp,float plateau,float h,float lsd_ref) {
    float left=__shfl_sync(mask,v,2,4),right=__shfl_sync(mask,v,3,4);
    float drive=0;
    if(lane>=2) {
        float wm=0.5f*m.gear*(left+right), r=fabsf(wm)*9.54929658551372f;
        float torque=r>rpm[nm-1]?0.0f:lookup(r,rpm,trq,nm);
        torque=fminf(torque,m.demand/fmaxf(fabsf(wm),1.f));
        torque=fminf(torque,lookup(fminf(fmaxf(r,brpm[0]),brpm[nb-1]),brpm,btrq,nb)*m.battery);
        torque=m.state+m.alpha*(torque-m.state);
        float axle=(torque-m.coast*tanhf((m.gear*wm)*(1.f/30.f)))*m.gear*m.eff;
        float dn=(left-right)*9.54929658551372f;
        float mag=m.torque_ref*powf(fabsf(dn)*(1.f/lsd_ref),m.exponent);
        float coupling=fminf(m.cap,fmaxf(-m.cap,0.5f*((dn>0)-(dn<0))*mag));
        drive=lane==2?axle*0.5f-coupling:axle*0.5f+coupling;
    }
    float sx=(w.vx-v*w.radius)/w.denom, ax=fabsf(sx), zx=ax*(1.f/kp);
    float rn=fmaxf(hypotf(zx,w.zy),1e-9f),rt=fmaxf(hypotf(ax,w.sy),1e-9f);
    float scale=((sx<0)-(sx>0))*w.peak*w.fz;
    float fn=scale*shape(fminf(rn*kp,plateau),w)*zx/rn;
    float ft=scale*shape(fminf(rt,plateau),w)*ax/rt;
    float fx=(1.f-w.blend)*fn+w.blend*ft;
    return w.inertia*(v-w.old)-h*(drive-w.resist-w.radius*fx);
}
extern "C" __global__ void solve(const float* wheels,const float* motors,const float* rpm,const float* trq,
 const float* brpm,const float* btrq,const bool* locked,float* out,int n,int nm,int nb,int expand,
 int bisect,int corrections,float kp,float plateau,float h,float lsd_ref) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n*4)return;
    unsigned mask=__activemask();
    int lane=i%4, car=i/4, stride=4*n;
    Wheel w={wheels[i],wheels[stride+i],wheels[2*stride+i],wheels[3*stride+i],wheels[4*stride+i],
    wheels[5*stride+i],wheels[6*stride+i],wheels[7*stride+i],wheels[8*stride+i],wheels[9*stride+i],
    wheels[10*stride+i],wheels[11*stride+i],wheels[12*stride+i],wheels[13*stride+i]};
    Motor m={motors[car],motors[n+car],motors[2*n+car],motors[3*n+car],motors[4*n+car],motors[5*n+car],
    motors[6*n+car],motors[7*n+car],motors[8*n+car],motors[9*n+car]};
    float lo=0.f,hi=fmaxf(10.f,2.f*w.old+10.f);
    #define G(v) residual(v,w,m,lane,mask,rpm,trq,nm,brpm,btrq,nb,kp,plateau,h,lsd_ref)
    for(int k=0;k<expand;k++){
        bool need=G(hi)<0;
        if(!__any_sync(mask,need))break;
        if(need)hi=2.f*hi+10.f;
    }
    for(int k=0;k<bisect;k++){
        float mid=0.5f*(lo+hi);
        if(G(mid)<0)lo=mid;else hi=mid;
    }

    float v=locked[i]?0.f:0.5f*(lo+hi);
    for(int k=0;k<corrections;k++){
        float g=G(v), jl=(G(v+(lane%2==0?0.05f:0.f))-g)*20.f;
        float jr=(G(v+(lane%2==1?0.05f:0.f))-g)*20.f;
        float a=__shfl_sync(mask,jl,2,4), b=__shfl_sync(mask,jr,2,4);
        float c=__shfl_sync(mask,jl,3,4), d=__shfl_sync(mask,jr,3,4);
        float l=__shfl_sync(mask,g,2,4),r=__shfl_sync(mask,g,3,4);
        float det=fmaxf(a*d-b*c,1e-4f);
        float corr=lane==0?g/fmaxf(jl,1e-4f):lane==1?g/fmaxf(jr,1e-4f):
                   lane==2?(d*l-b*r)/det:(a*r-c*l)/det;
        v=fmaxf(0.f,v-fminf(5.f,fmaxf(-5.f,corr)));
    }
    out[i]=v;
}

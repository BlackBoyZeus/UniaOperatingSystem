import { injectable } from 'inversify';
import { controller, httpPost, httpGet, httpPut, httpDelete, authorize } from 'inversify-express-utils';
import { Request, Response, NextFunction } from 'express';
import { Logger } from 'winston';
import { RateLimit } from 'express-rate-limit';

import { FleetService } from '../../services/fleet/FleetService';
import { 
    IFleet, 
    IFleetMember, 
    FleetStatus, 
    FleetMetrics, 
    FleetHealth 
} from '../../interfaces/fleet.interface';
import { 
    validateFleetCreation, 
    validateFleetUpdate, 
    validateFleetMetrics 
} from '../validators/fleet.validator';

// Constants for fleet management
const MAX_FLEET_SIZE = 32;
const MAX_LATENCY_THRESHOLD = 50;
const REQUEST_TIMEOUT = 5000;
const RATE_LIMIT_WINDOW = 60000;
const RATE_LIMIT_MAX = 100;

@injectable()
@controller('/api/fleet')
@RateLimit({ 
    windowMs: RATE_LIMIT_WINDOW,
    max: RATE_LIMIT_MAX,
    message: 'Too many requests from this IP'
})
export class FleetController {
    private readonly correlationIdKey: string = 'x-correlation-id';

    constructor(
        private readonly fleetService: FleetService,
        private readonly logger: Logger
    ) {}

    @httpPost('/')
    @authorize('fleet:create')
    public async createFleet(req: Request, res: Response, next: NextFunction): Promise<Response> {
        const correlationId = req.get(this.correlationIdKey) || Date.now().toString();
        
        try {
            this.logger.info('Creating new fleet', { 
                correlationId,
                requestBody: req.body 
            });

            await validateFleetCreation(req.body);

            const fleet = await this.fleetService.createFleet(req.body);

            this.logger.info('Fleet created successfully', {
                correlationId,
                fleetId: fleet.id
            });

            return res.status(201).json({
                success: true,
                data: fleet,
                correlationId
            });

        } catch (error) {
            this.logger.error('Fleet creation failed', {
                correlationId,
                error: error.message,
                stack: error.stack
            });

            return res.status(400).json({
                success: false,
                error: error.message,
                correlationId
            });
        }
    }

    @httpPost('/:fleetId/join')
    @authorize('fleet:join')
    public async joinFleet(req: Request, res: Response, next: NextFunction): Promise<Response> {
        const correlationId = req.get(this.correlationIdKey) || Date.now().toString();
        const { fleetId } = req.params;
        
        try {
            this.logger.info('Processing fleet join request', {
                correlationId,
                fleetId,
                memberId: req.body.id
            });

            await this.fleetService.joinFleet(fleetId, req.body);

            this.logger.info('Member joined fleet successfully', {
                correlationId,
                fleetId,
                memberId: req.body.id
            });

            return res.status(200).json({
                success: true,
                correlationId
            });

        } catch (error) {
            this.logger.error('Fleet join failed', {
                correlationId,
                fleetId,
                error: error.message
            });

            return res.status(400).json({
                success: false,
                error: error.message,
                correlationId
            });
        }
    }

    @httpDelete('/:fleetId/members/:memberId')
    @authorize('fleet:leave')
    public async leaveFleet(req: Request, res: Response, next: NextFunction): Promise<Response> {
        const correlationId = req.get(this.correlationIdKey) || Date.now().toString();
        const { fleetId, memberId } = req.params;

        try {
            this.logger.info('Processing fleet leave request', {
                correlationId,
                fleetId,
                memberId
            });

            await this.fleetService.leaveFleet(fleetId, memberId);

            this.logger.info('Member left fleet successfully', {
                correlationId,
                fleetId,
                memberId
            });

            return res.status(200).json({
                success: true,
                correlationId
            });

        } catch (error) {
            this.logger.error('Fleet leave failed', {
                correlationId,
                error: error.message
            });

            return res.status(400).json({
                success: false,
                error: error.message,
                correlationId
            });
        }
    }

    @httpDelete('/:fleetId')
    @authorize('fleet:disband')
    public async disbandFleet(req: Request, res: Response, next: NextFunction): Promise<Response> {
        const correlationId = req.get(this.correlationIdKey) || Date.now().toString();
        const { fleetId } = req.params;

        try {
            this.logger.info('Processing fleet disband request', {
                correlationId,
                fleetId
            });

            await this.fleetService.disbandFleet(fleetId);

            this.logger.info('Fleet disbanded successfully', {
                correlationId,
                fleetId
            });

            return res.status(200).json({
                success: true,
                correlationId
            });

        } catch (error) {
            this.logger.error('Fleet disband failed', {
                correlationId,
                error: error.message
            });

            return res.status(400).json({
                success: false,
                error: error.message,
                correlationId
            });
        }
    }

    @httpGet('/:fleetId/health')
    @authorize('fleet:monitor')
    public async getFleetHealth(req: Request, res: Response, next: NextFunction): Promise<Response> {
        const correlationId = req.get(this.correlationIdKey) || Date.now().toString();
        const { fleetId } = req.params;

        try {
            const health = await this.fleetService.getFleetHealth(fleetId);

            if (health.averageLatency > MAX_LATENCY_THRESHOLD) {
                this.logger.warn('High fleet latency detected', {
                    correlationId,
                    fleetId,
                    latency: health.averageLatency
                });
            }

            return res.status(200).json({
                success: true,
                data: health,
                correlationId
            });

        } catch (error) {
            this.logger.error('Fleet health check failed', {
                correlationId,
                error: error.message
            });

            return res.status(500).json({
                success: false,
                error: error.message,
                correlationId
            });
        }
    }

    @httpGet('/active')
    @authorize('fleet:list')
    public async getActiveFleets(req: Request, res: Response, next: NextFunction): Promise<Response> {
        const correlationId = req.get(this.correlationIdKey) || Date.now().toString();

        try {
            const fleets = await this.fleetService.getActiveFleets();

            return res.status(200).json({
                success: true,
                data: fleets,
                correlationId
            });

        } catch (error) {
            this.logger.error('Active fleets retrieval failed', {
                correlationId,
                error: error.message
            });

            return res.status(500).json({
                success: false,
                error: error.message,
                correlationId
            });
        }
    }

    @httpGet('/:fleetId/metrics')
    @authorize('fleet:monitor')
    public async getFleetMetrics(req: Request, res: Response, next: NextFunction): Promise<Response> {
        const correlationId = req.get(this.correlationIdKey) || Date.now().toString();
        const { fleetId } = req.params;

        try {
            const metrics = await this.fleetService.getFleetMetrics(fleetId);

            return res.status(200).json({
                success: true,
                data: metrics,
                correlationId
            });

        } catch (error) {
            this.logger.error('Fleet metrics retrieval failed', {
                correlationId,
                error: error.message
            });

            return res.status(500).json({
                success: false,
                error: error.message,
                correlationId
            });
        }
    }
}